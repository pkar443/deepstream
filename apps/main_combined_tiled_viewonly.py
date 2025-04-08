import sys
import gi
import pyds
from gi.repository import Gst, GObject, GLib
import os
import time
from collections import defaultdict

# Ensure correct version of GStreamer
gi.require_version('Gst', '1.0')

# Dictionary to hold FPS info
frame_times = defaultdict(list)

# === Probe to Parse DeepStream Metadata ===
def osd_sink_pad_buffer_probe(pad, info, u_data):
    buffer = info.get_buffer()
    if not buffer:
        print("Unable to get GstBuffer!")
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    if not batch_meta:
        print("No batch metadata found!")
        return Gst.PadProbeReturn.OK

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            if not frame_meta:
                continue

            stream_id = frame_meta.pad_index
            current_time = time.time()
            frame_times[stream_id].append(current_time)
            if len(frame_times[stream_id]) > 30:
                frame_times[stream_id] = frame_times[stream_id][-30:]

            if len(frame_times[stream_id]) > 1:
                time_diff = frame_times[stream_id][-1] - frame_times[stream_id][0]
                fps = len(frame_times[stream_id]) / time_diff if time_diff > 0 else 0.0
            else:
                fps = 0.0

            # Set display_text for the frame using display_meta
            display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
            display_meta.num_labels = 1
            display_meta.text_params[0].display_text = f"FPS: {fps:.1f}"
            display_meta.text_params[0].x_offset = 10
            display_meta.text_params[0].y_offset = 20
            display_meta.text_params[0].font_params.font_color.set(1.0, 1.0, 1.0, 1.0)
            display_meta.text_params[0].font_params.font_size = 16
            display_meta.text_params[0].font_params.font_name = "Serif"
            pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

            print(f"✅ Stream {stream_id} | Frame {frame_meta.frame_num} | FPS: {fps:.2f}")

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                if not obj_meta:
                    continue

                label = ""
                l_class = obj_meta.classifier_meta_list
                while l_class is not None:
                    class_meta = pyds.NvDsClassifierMeta.cast(l_class.data)
                    l_label = class_meta.label_info_list
                    while l_label is not None:
                        label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                        if label_info.result_label:
                            prob = max(0.0, min(label_info.result_prob, 1.0))
                            obj_meta.text_params.display_text = f"{label_info.result_label} ({prob:.2f})"
                            label = label_info.result_label
                        l_label = l_label.next
                    l_class = l_class.next

                print(
                    f"✅ Obj Class: {obj_meta.class_id}, "
                    f"Confidence: {obj_meta.confidence:.2f}, "
                    f"BBox: [{obj_meta.rect_params.left:.2f}, {obj_meta.rect_params.top:.2f}, "
                    f"{obj_meta.rect_params.width:.2f}, {obj_meta.rect_params.height:.2f}] {label}"
                )

                l_obj = l_obj.next

        except Exception as e:
            print(f"❗️ Error processing frame metadata: {e}")
            continue

        l_frame = l_frame.next

    return Gst.PadProbeReturn.OK

def calculate_tiler_dims(num_sources):
    import math
    rows = int(math.sqrt(num_sources))
    columns = int(math.ceil(num_sources / rows))
    return rows, columns

def main():
    Gst.init(None)

    # Load video file paths
    with open("/workspace/deepstream-app/configs/input_sources.txt") as f:
        video_paths = [line.strip() for line in f.readlines() if line.strip()]

    num_sources = len(video_paths)
    rows, cols = calculate_tiler_dims(num_sources)

    # Create pipeline and elements
    pipeline = Gst.Pipeline()
    streammux = Gst.ElementFactory.make("nvstreammux", "mux")
    streammux.set_property("batch-size", num_sources)
    streammux.set_property("width", 640)
    streammux.set_property("height", 640)
    streammux.set_property("live-source", 0)
    streammux.set_property("sync-inputs", 0)

    pipeline.add(streammux)

    for i, path in enumerate(video_paths):
        uridecodebin = Gst.ElementFactory.make("uridecodebin", f"source{i}")
        uridecodebin.set_property("uri", f"file://{path}")
        uridecodebin.connect("pad-added", lambda bin, pad, i=i: pad.link(streammux.get_request_pad(f"sink_{i}")))
        pipeline.add(uridecodebin)

    primary_infer = Gst.ElementFactory.make("nvinfer", "primary-infer")
    primary_infer.set_property("config-file-path", "/workspace/deepstream-app/configs/config_infer_primary.txt")

    secondary_infer = Gst.ElementFactory.make("nvinfer", "secondary-infer")
    secondary_infer.set_property("config-file-path", "/workspace/deepstream-app/configs/config_infer_secondary.txt")

    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "tiler")
    tiler.set_property("rows", rows)
    tiler.set_property("columns", cols)
    tiler.set_property("width", 1280)
    tiler.set_property("height", 720)

    nvdsosd = Gst.ElementFactory.make("nvdsosd", "osd")
    nvdsosd.set_property("display-clock", 0)  # Turn off default clock overlay

    videoconvert = Gst.ElementFactory.make("nvvideoconvert", "convert")
    sink = Gst.ElementFactory.make("nveglglessink", "sink")

    for elem in [primary_infer, secondary_infer, tiler, nvdsosd, videoconvert, sink]:
        pipeline.add(elem)

    streammux.link(primary_infer)
    primary_infer.link(secondary_infer)
    secondary_infer.link(tiler)
    tiler.link(nvdsosd)
    nvdsosd.link(videoconvert)
    videoconvert.link(sink)

    osd_sink_pad = nvdsosd.get_static_pad("sink")
    if osd_sink_pad:
        osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    print("✅ Running DeepStream Batch Tiled Pipeline with Real FPS Overlay...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop = GLib.MainLoop()
        loop.run()
    except KeyboardInterrupt:
        print("❗️ Interrupted. Stopping pipeline...")
        pipeline.set_state(Gst.State.NULL)
        loop.quit()

if __name__ == "__main__":
    sys.exit(main())
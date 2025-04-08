import sys
import gi
import pyds  # For accessing DeepStream metadata
from gi.repository import Gst, GObject, GLib

gi.require_version('Gst', '1.0')

# === Probe to Parse DeepStream Metadata ===
def osd_sink_pad_buffer_probe(pad, info, u_data):
    buffer = info.get_buffer()
    if not buffer:
        print("Unable to get GstBuffer!")
        return Gst.PadProbeReturn.OK

    print(">>> Processing new frame...")

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    if not batch_meta:
        print("No batch metadata found!")
        return Gst.PadProbeReturn.OK

    caps = pad.get_current_caps()
    structure = caps.get_structure(0)
    model_width = structure.get_int("width")[1]
    model_height = structure.get_int("height")[1]

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            if not frame_meta:
                continue

            print(f"✅ Processing Frame ID: {frame_meta.frame_num}")
            original_width = frame_meta.source_frame_width
            original_height = frame_meta.source_frame_height
            scale_x = original_width / model_width
            scale_y = original_height / model_height

            print(f"🔎 Scale factors - X: {scale_x:.4f}, Y: {scale_y:.4f}")

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                if not obj_meta:
                    continue

                obj_meta.rect_params.left *= scale_x
                obj_meta.rect_params.top *= scale_y
                obj_meta.rect_params.width *= scale_x
                obj_meta.rect_params.height *= scale_y

                obj_meta.rect_params.left = max(0, min(obj_meta.rect_params.left, original_width - 1))
                obj_meta.rect_params.top = max(0, min(obj_meta.rect_params.top, original_height - 1))
                obj_meta.rect_params.width = max(0, min(obj_meta.rect_params.width, original_width - obj_meta.rect_params.left))
                obj_meta.rect_params.height = max(0, min(obj_meta.rect_params.height, original_height - obj_meta.rect_params.top))

                label = ""
                l_class = obj_meta.classifier_meta_list
                while l_class is not None:
                    class_meta = pyds.NvDsClassifierMeta.cast(l_class.data)
                    l_label = class_meta.label_info_list
                    while l_label is not None:
                        label_info = pyds.NvDsLabelInfo.cast(l_label.data)
                        if label_info.result_label:
                            # Fix: Clamp the probability to [0, 1] to avoid abnormal values like 222.65
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


def main():
    Gst.init(None)
    pipeline = Gst.parse_launch(
        "filesrc location=/workspace/deepstream-videos/event20241220082716118.avi "
        "! avidemux ! h264parse ! nvv4l2decoder ! nvvideoconvert "
        "! video/x-raw(memory:NVMM), format=RGBA, width=640, height=640 "
        "! mux.sink_0 "
        "nvstreammux name=mux batch-size=1 width=640 height=640 live-source=0 sync-inputs=0 "
        "! nvinfer config-file-path=/workspace/deepstream-app/configs/config_infer_primary.txt name=primary-infer "
        "! nvinfer config-file-path=/workspace/deepstream-app/configs/config_infer_secondary.txt name=secondary-infer "
        "! nvdsosd name=osd ! nvvideoconvert ! nvv4l2h264enc bitrate=8000000 ! h264parse ! qtmux "
        "! filesink location=/workspace/deepstream-videos/output_video.mp4"
    )

    osd_element = pipeline.get_by_name("osd")
    if not osd_element:
        print("Could not get nvdsosd element!")
        sys.exit(1)

    osd_sink_pad = osd_element.get_static_pad("sink")
    if not osd_sink_pad:
        print("Unable to get sink pad of OSD!")
        sys.exit(1)

    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    print("✅ Running DeepStream Inference with Secondary GPU Classifier...")
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

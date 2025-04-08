#include "nvdsinfer_custom_impl.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>

#define NUM_CLASSES 3

extern "C" bool NvDsInferParseCustomMMYOLO(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

static __inline__ float clamp(float val, float min, float max) {
    return val > min ? (val < max ? val : max) : min;
}

static float IoU(const NvDsInferParseObjectInfo& a, const NvDsInferParseObjectInfo& b) {
    float x1 = std::max(a.left, b.left);
    float y1 = std::max(a.top, b.top);
    float x2 = std::min(a.left + a.width, b.left + b.width);
    float y2 = std::min(a.top + a.height, b.top + b.height);

    float interArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float unionArea = a.width * a.height + b.width * b.height - interArea;

    return interArea / unionArea;
}

static std::vector<NvDsInferParseObjectInfo> applyNMS(
    std::vector<NvDsInferParseObjectInfo>& detections,
    float iouThreshold)
{
    std::sort(detections.begin(), detections.end(), [](const NvDsInferParseObjectInfo& a, const NvDsInferParseObjectInfo& b) {
        return a.detectionConfidence > b.detectionConfidence;
    });

    std::vector<NvDsInferParseObjectInfo> finalDetections;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        finalDetections.push_back(detections[i]);
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (!suppressed[j] && detections[i].classId == detections[j].classId && IoU(detections[i], detections[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }

    return finalDetections;
}

static std::vector<NvDsInferParseObjectInfo> decodeMMYoloTensor(
    const float* bboxes,
    const float* scores,
    const int* labels,
    const unsigned int& num_dets,
    const NvDsInferParseDetectionParams& detectionParams,
    const unsigned int& img_w,
    const unsigned int& img_h)
{
    std::vector<NvDsInferParseObjectInfo> bboxInfo;
    std::map<int, int> class_counts;

    for (unsigned int i = 0; i < num_dets; ++i) {
        int cls_id = std::max(0, std::min(labels[i], NUM_CLASSES - 1));
        float conf_thres = detectionParams.perClassThreshold[cls_id];
        float score = scores[i];

        float x0 = bboxes[i * 4];
        float y0 = bboxes[i * 4 + 1];
        float x1 = bboxes[i * 4 + 2];
        float y1 = bboxes[i * 4 + 3];
        float width = x1 - x0;
        float height = y1 - y0;

        std::cout << "[DEBUG] Candidate BBox - Class: " << cls_id
                  << ", Score: " << score
                  << ", W: " << width << ", H: " << height << std::endl;

        // Temporarily disable size-based filtering for debugging
        // bool tooSmall = (width < 0.01f * img_w || height < 0.01f * img_h);
        // bool tooBig = (width > img_w || height > img_h);
        // if ((tooSmall && score < 0.7f) || tooBig) continue;

        if (score < conf_thres) continue;

        x0 = clamp(x0, 0.f, img_w);
        y0 = clamp(y0, 0.f, img_h);
        x1 = clamp(x1, 0.f, img_w);
        y1 = clamp(y1, 0.f, img_h);

        NvDsInferParseObjectInfo obj;
        obj.left = x0;
        obj.top = y0;
        obj.width = width;
        obj.height = height;
        obj.detectionConfidence = score;
        obj.classId = cls_id;
        bboxInfo.push_back(obj);

        class_counts[cls_id]++;
    }

    std::cout << "Label distribution: ";
    for (int i = 0; i < NUM_CLASSES; ++i) {
        std::cout << "Class " << i << ": " << class_counts[i] << " | ";
    }
    std::cout << std::endl;

    return bboxInfo;
}

extern "C" bool NvDsInferParseCustomMMYOLO(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    std::cout << " Output Layers Info Dump:" << std::endl;
    for (size_t i = 0; i < outputLayersInfo.size(); ++i) {
        std::cout << "  Layer " << i << ": " << outputLayersInfo[i].layerName
                  << " | DataType: " << outputLayersInfo[i].dataType
                  << " | dims (" << outputLayersInfo[i].inferDims.numDims << "): [";

        for (unsigned int j = 0; j < outputLayersInfo[i].inferDims.numDims; ++j)
            std::cout << outputLayersInfo[i].inferDims.d[j] << (j < outputLayersInfo[i].inferDims.numDims - 1 ? ", " : "");

        std::cout << "]" << std::endl;
    }

    if (outputLayersInfo.size() < 2) {
        std::cerr << "Error: insufficient output layers for MMYOLO" << std::endl;
        return false;
    }

    const float* dets = static_cast<const float*>(outputLayersInfo[0].buffer);
    const int* labels = static_cast<const int*>(outputLayersInfo[1].buffer);
    unsigned int num_dets = outputLayersInfo[0].inferDims.d[0];

    if (!dets || !labels || num_dets == 0) {
        std::cerr << "[WARN] No detections found in the buffer." << std::endl;
        return true;
    }

    std::vector<float> bboxes(num_dets * 4);
    std::vector<float> scores(num_dets);

    for (unsigned int i = 0; i < num_dets; ++i) {
        bboxes[i * 4 + 0] = dets[i * 5 + 0];
        bboxes[i * 4 + 1] = dets[i * 5 + 1];
        bboxes[i * 4 + 2] = dets[i * 5 + 2];
        bboxes[i * 4 + 3] = dets[i * 5 + 3];
        scores[i] = dets[i * 5 + 4];
    }

    std::vector<NvDsInferParseObjectInfo> rawDetections = decodeMMYoloTensor(
        bboxes.data(),
        scores.data(),
        labels,
        num_dets,
        detectionParams,
        networkInfo.width,
        networkInfo.height
    );

    std::vector<NvDsInferParseObjectInfo> finalObjects;
    for (unsigned int cls = 0; cls < NUM_CLASSES; ++cls) {
        std::vector<NvDsInferParseObjectInfo> clsObjects;
        for (const auto& obj : rawDetections) {
            if (static_cast<unsigned int>(obj.classId) == cls) clsObjects.push_back(obj);
        }
        auto nmsed = applyNMS(clsObjects, 0.3f); // IoU threshold = 0.3
        finalObjects.insert(finalObjects.end(), nmsed.begin(), nmsed.end());
    }

    objectList = finalObjects;

    std::cout << "[DEBUG] Total Objects Parsed: " << objectList.size() << std::endl;
    for (size_t i = 0; i < objectList.size(); ++i) {
        const auto& obj = objectList[i];
        std::cout << "[DEBUG] Obj " << i
                  << " | ClassID: " << obj.classId
                  << " | Conf: " << obj.detectionConfidence
                  << " | BBox: [" << obj.left << ", " << obj.top
                  << ", " << obj.width << ", " << obj.height << "]"
                  << std::endl;
    }

    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomMMYOLO);

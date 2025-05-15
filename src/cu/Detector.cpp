#include "detector_dl/Detector.hpp"

Detector::Detector(const int &NUM_CLASSES, const std::string TARGET_COLOUR, const float &NMS_THRESH, const float &BBOX_CONF_THRESH, 
                   const int &INPUT_W, const int &INPUT_H, const std::string engine_file_path, const ArmorParams &a)
             : NUM_CLASSES(NUM_CLASSES), TARGET_COLOUR(TARGET_COLOUR), NMS_THRESH(NMS_THRESH), BBOX_CONF_THRESH(BBOX_CONF_THRESH), 
               INPUT_W(INPUT_W), INPUT_H(INPUT_H), engine_file_path(engine_file_path), a(a)
{   
    Detector::InitModelEngine();
    Detector::AllocMem();
}

void Detector::InitModelEngine()
{
    cudaSetDevice(DEVICE);
    std::ifstream file {this->engine_file_path, std::ios::binary};
    size_t size {0};
    char *trtModelStreamDet {nullptr};
    if (file.good())
    {
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trtModelStreamDet = new char[size];
    assert(trtModelStreamDet);
    file.read(trtModelStreamDet, size);
    file.close();
    }
    runtime_det = createInferRuntime(gLogger);
    assert(runtime_det != nullptr);
    engine_det = runtime_det->deserializeCudaEngine(trtModelStreamDet, size);
    assert(engine_det != nullptr);
    this->context_det = engine_det->createExecutionContext();
    assert(context_det != nullptr);
    delete[] trtModelStreamDet;
}


void Detector::AllocMem()
{
    inputIndex = engine_det->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine_det->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);

    auto out_dims = engine_det->getBindingDimensions(1);
    auto output_size = 1;
    OUTPUT_CANDIDATES = out_dims.d[1];
    for (int i = 0; i < out_dims.nbDims; ++i)
    {
        output_size *= out_dims.d[i];
    }

    // 尝试优化:
    CHECK(cudaMalloc(&buffers[inputIndex], sizeof(float) * (3 * INPUT_H * INPUT_W)));
    CHECK(cudaMalloc(&buffers[outputIndex], sizeof(float) * output_size));
    // CHECK(cudaMallocManaged(&buffers[inputIndex], sizeof(float) * (3 * INPUT_H * INPUT_W)));
    // CHECK(cudaMallocManaged(&buffers[outputIndex], sizeof(float) * output_size));

    CHECK(cudaMallocHost(&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    CHECK(cudaMalloc(&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
    //---------------------------------------------------------------------------
    // CHECK(cudaMallocManaged(&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3, cudaMemAttachHost));
    // CHECK(cudaMallocManaged(&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));  

    CHECK(cudaMallocHost(&affine_matrix_d2i_host, sizeof(float) * 6));
    CHECK(cudaMalloc(&affine_matrix_d2i_device, sizeof(float) * 6));
    //---------------------------------------------------------------------------
    // CHECK(cudaMallocManaged(&affine_matrix_d2i_device, sizeof(float) * 6));


    CHECK(cudaMalloc(&decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT)));
    // CHECK(cudaMallocManaged(&decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT)));
}

std::vector<Armor> Detector::detect(cv::Mat &frame, bool show_img = false)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 预处理
    AffineMatrix afmt;
    // CHECK(cudaMallocManaged(&(afmt.d2i), sizeof(float) * 6, cudaMemAttachHost));
    getd2i(afmt, {INPUT_W, INPUT_H}, cv::Size(frame.cols, frame.rows)); // TODO
    float *buffer_idx = (float *)buffers[inputIndex];
    size_t img_size = frame.cols * frame.rows * 3;

    memcpy(affine_matrix_d2i_host, afmt.d2i, sizeof(afmt.d2i));
    CHECK(cudaMemcpyAsync(affine_matrix_d2i_device, affine_matrix_d2i_host, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream));
    // CHECK(cudaStreamAttachMemAsync(stream, afmt.d2i, 0, cudaMemAttachGlobal));
    // CHECK(cudaMemcpyAsync(affine_matrix_d2i_device, afmt.d2i, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream));

    memcpy(img_host, frame.data, img_size);
    CHECK(cudaMemcpyAsync(img_device, img_host, img_size, cudaMemcpyHostToDevice, stream));
/*         CHECK(cudaMallocManaged(&(this->frame.data), img_size, cudaMemAttachHost));
    CHECK(cudaStreamAttachMemAsync(stream, (this->frame.data), 0, cudaMemAttachGlobal)); */
    // CHECK(cudaMemcpyAsync(img_device, this->frame.data, img_size, cudaMemcpyHostToDevice, stream));
    preprocess_kernel_img(img_device, frame.cols, frame.rows, 
        buffer_idx, INPUT_W, INPUT_H, 
        affine_matrix_d2i_device, stream);
    // 推理
    (*context_det).enqueueV2((void **)buffers, stream, nullptr);
    float *predict = (float *)buffers[outputIndex];
    // 后处理
    CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(int), stream));
    decode_kernel_invoker(
        predict, NUM_BOX_ELEMENT, OUTPUT_CANDIDATES, NUM_CLASSES, 
        CKPT_NUM, BBOX_CONF_THRESH, affine_matrix_d2i_device, 
        decode_ptr_device, MAX_OBJECTS, stream);
    nms_kernel_invoker(decode_ptr_device, NMS_THRESH, MAX_OBJECTS, stream, NUM_BOX_ELEMENT);    
    CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT), cudaMemcpyDeviceToHost, stream));
    // CHECK(cudaStreamAttachMemAsync(stream, decode_ptr_device, 0, cudaMemAttachHost));
    cudaStreamSynchronize(stream);

    std::vector<Armor> armors;
    std::vector<bbox> boxes;
    int boxes_count = 0;
    int count = std::min((int)*decode_ptr_host, MAX_OBJECTS);
    bool detected = true;
    this->is_detected = (count !=0) ? true : false;
    if (this->is_detected)
    {
        detected = false;
        for (int i = 0; i < count; ++i)
        {
            int basic_pos = 1 + i * NUM_BOX_ELEMENT;
            int keep_flag = decode_ptr_host[basic_pos + 6];
            if (keep_flag == 1)
            {
                boxes_count += 1;
                bbox box;
                box.x1 = decode_ptr_host[basic_pos + 0];
                box.y1 = decode_ptr_host[basic_pos + 1];
                box.x2 = decode_ptr_host[basic_pos + 2];
                box.y2 = decode_ptr_host[basic_pos + 3];
                box.score = decode_ptr_host[basic_pos + 4];
                box.class_id = decode_ptr_host[basic_pos + 5];

                int landmark_pos = basic_pos + 7;
                for(int id = 0; id < CKPT_NUM; id += 1)
                {
                    box.landmarks[2 * id] = decode_ptr_host[landmark_pos + 2 * id];
                    box.landmarks[2 * id + 1] = decode_ptr_host[landmark_pos + 2 * id + 1];
                }
                boxes.push_back(box);
            }
        }
        for (auto box : boxes)
        {
            // bl->tl->tr->br 左下 左上 右上 右下
            std::vector<cv::Point2f> points{cv::Point2f(box.landmarks[0], box.landmarks[1]),
                                            cv::Point2f(box.landmarks[2], box.landmarks[3]),
                                            cv::Point2f(box.landmarks[4], box.landmarks[5]),
                                            cv::Point2f(box.landmarks[6], box.landmarks[7])};
            Armor armor = Armor{points};

            float light_left_length = abs(box.landmarks[1] - box.landmarks[3]);
            float light_right_length = abs(box.landmarks[7] - box.landmarks[5]);
            float avg_light_length = (light_left_length + light_right_length) / 2;
            // std::cout << "left_proportion: " << 
            cv::Point2f light_left_center = cv::Point2f((box.landmarks[2] + box.landmarks[0]) / 2, (box.landmarks[3] + box.landmarks[1]) / 2);
            cv::Point2f light_right_center = cv::Point2f((box.landmarks[6] + box.landmarks[4]) / 2, (box.landmarks[7] + box.landmarks[5]) / 2);
            float center_distance = cv::norm(light_left_center - light_right_center) / avg_light_length;
            
            if(this->TARGET_COLOUR == "BLUE" && box.class_id <=8 && box.class_id >=0 )
            {
                armor.type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
                if(box.class_id == 0)
                    armor.number = "guard";
                else if(box.class_id >= 1 && box.class_id <= 5)
                    armor.number = std::to_string(box.class_id);
                else if(box.class_id == 6)
                    armor.number = "outpost";
                else if(box.class_id == 7||box.class_id == 8)
                    armor.number = "base";

                detected = true;
                armors.emplace_back(armor);

            }
            else if(this->TARGET_COLOUR == "RED" && box.class_id <=17 && box.class_id >=9 )
            {
                armor.type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
                if(box.class_id == 9)
                    armor.number = "guard";
                else if(box.class_id >= 10 && box.class_id <= 14)
                    armor.number = std::to_string(box.class_id - 9);
                else if(box.class_id == 15)
                    armor.number = "outpost";
                else if(box.class_id == 16||box.class_id == 17)
                    armor.number = "base";

                detected = true;
                armors.emplace_back(armor);
            }
            else
            {
                armor.number = "negative";
            }

            
        }

        if (show_img && detected)
        {
            std::cout << "Detected Armor!" << std::endl;
            for (int i = 0; i < boxes_count; i++)
            {
                if((this->TARGET_COLOUR == "BLUE" && boxes[i].class_id <=8 && boxes[i].class_id >=0) ||  (this->TARGET_COLOUR == "RED" && boxes[i].class_id <=17 && boxes[i].class_id >=9))
                {
                    for (int j = 0; j < CKPT_NUM; j++)
                    {
                        cv::Scalar color = cv::Scalar(color_list[j][0], color_list[j][1], color_list[j][2]);
                        cv::circle(frame, cv::Point(boxes[i].landmarks[2 * j], boxes[i].landmarks[2 * j + 1]), 2, color, -1);
                    }

                    std::string label = std::to_string(boxes[i].class_id);
                    cv::putText(frame, label, cv::Point(boxes[i].x1, boxes[i].y1 - 2), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 2);

                    cv::line(frame,
                            cv::Point(boxes[i].landmarks[0], boxes[i].landmarks[1]),
                            cv::Point(boxes[i].landmarks[4], boxes[i].landmarks[5]),
                            cv::Scalar(0, 255, 0),
                            2);
                    cv::line(frame,
                            cv::Point(boxes[i].landmarks[2], boxes[i].landmarks[3]),
                            cv::Point(boxes[i].landmarks[6], boxes[i].landmarks[7]),
                            cv::Scalar(0, 255, 0),
                            2);
                    cv::line(frame,
                            cv::Point(boxes[i].landmarks[0], boxes[i].landmarks[1]),
                            cv::Point(boxes[i].landmarks[2], boxes[i].landmarks[3]),
                            cv::Scalar(0, 255, 0),
                            2);
                    cv::line(frame,
                            cv::Point(boxes[i].landmarks[4], boxes[i].landmarks[5]),
                            cv::Point(boxes[i].landmarks[6], boxes[i].landmarks[7]),
                            cv::Scalar(0, 255, 0),
                            2);
                }

            }
        }
        else
        {
            cv::putText(frame, "No Detected!", cv::Point2f{100, 100}, cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 2);
        }

    }
    cudaStreamDestroy(stream);
    return armors;
}

void Detector::Release()
    {
        context_det->destroy();
        engine_det->destroy();
        runtime_det->destroy();

        cudaStreamDestroy(stream);
        CHECK(cudaFree(affine_matrix_d2i_device));
        CHECK(cudaFreeHost(affine_matrix_d2i_host));
        CHECK(cudaFree(img_device));
        CHECK(cudaFreeHost(img_host));
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));
        CHECK(cudaFree(decode_ptr_device));
        delete[] decode_ptr_host;
        //decode_ptr_host = nullptr;
    }

Detector::~Detector()
{
    Detector::Release();
}
#include "darknet.h"
#include "yolo_classifier_class.hpp"

#include "network.h"
extern "C" {
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include "option_list.h"
#include "stb_image.h"
}

#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>


static std::unique_ptr<Classifier> classifier;

LIB_API int init_classifier(const char* data_filename, const char* cfg_filename, const char* weight_filename)
{
    classifier.reset(new Classifier(data_filename, cfg_filename, weight_filename));
    return 1;
}

LIB_API int predict_top_classifier(const char* image_filename, classifier_t_container& result, int top)
{
    std::vector<classifier_t> predictions_vec = classifier->predict(image_filename, top);

    for (size_t i = 0; i < predictions_vec.size() && i < C_SHARP_MAX_OBJECTS; ++i) {
        result.candidates[i] = predictions_vec[i];
    }
        
    return predictions_vec.size();
}

LIB_API size_t get_class_name(int class_id, char* result)
{
    strcpy(result, classifier->get_class_name(class_id));
    return strlen(result);
}

int dispose_classifier()
{
    classifier.reset();
    return 1;
}


LIB_API Classifier::Classifier(std::string data_filename, std::string cfg_filename, std::string weight_filename)
{
    _cfg_filename       = cfg_filename;
    _weight_filename    = weight_filename;
    _data_filename      = data_filename;

    char* cfgfile       = const_cast<char*>(_cfg_filename.c_str());
    char* weightfile    = const_cast<char*>(_weight_filename.c_str());
    char* datafile      = const_cast<char*>(_data_filename.c_str());

    net = parse_network_cfg_custom(cfgfile, 1, 0);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);

    list* options = read_data_cfg(datafile);

    char* name_list = option_find_str(options, "names", 0);
    if (!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    int classes = option_find_int(options, "classes", 2);
    classes_count = classes;

    names = get_labels(name_list);

    conf_top = option_find_int(options, "top", 1);
    indexes = (int*)xcalloc(conf_top, sizeof(int));

    printf(" classes = %d, output in cfg = %d, top = %d \n", classes, net.layers[net.n - 1].c, conf_top);
    fflush(stdout);

    free_list_contents_kvp(options);
    free_list(options);
}

LIB_API std::vector<classifier_t> Classifier::predict(std::string image_filename, int top)
{
    std::vector<classifier_t> result_vec;

    if (top != 0) conf_top = top;
    if (top > classes_count) conf_top = classes_count;

    char* imageFile = const_cast<char*>(image_filename.c_str());

    image im = load_image_color(imageFile, 0, 0);
    image resized = resize_min(im, net.w);
    image r = crop_image(resized, (resized.w - net.w) / 2, (resized.h - net.h) / 2, net.w, net.h);

    float* X = r.data;
    float* predictions = network_predict(net, X);

    if (net.hierarchy) hierarchy_predictions(predictions, net.outputs, net.hierarchy, 0);
    top_k(predictions, net.outputs, conf_top, indexes);
    
    for (int i = 0; i < conf_top; ++i) {
        int index = indexes[i];

        classifier_t ct;
        ct.class_id     = index;
        ct.prob         = predictions[index];

       /* printf(" f = %s, classid = %d, class_name = %s, prob = %f \n", imageFile, index, names[index], predictions[index]);
        fflush(stdout);*/

        result_vec.push_back(ct);
    }

    free_image(r);
    free_image(im);
    free_image(resized);
 
    return result_vec;
}

LIB_API char* Classifier::get_class_name(int class_id)
{
    if (names == nullptr)
        return (char*)"NONE";
    return names[class_id];
}

LIB_API Classifier::~Classifier()
{
    free_network(net);
}


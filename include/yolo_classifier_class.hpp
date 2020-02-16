#ifndef YOLO_CLASSIFIER_CLASS_HPP
#define YOLO_CLASSIFIER_CLASS_HPP

#define C_SHARP_MAX_OBJECTS 1000

struct classifier_t {
	unsigned int class_id;	// class id range [0, classes - 1]
	float prob;				// confidence - probability that the object was found correctly
};

struct classifier_t_container {
	classifier_t candidates[C_SHARP_MAX_OBJECTS];
};

#ifdef __cplusplus

#include <vector>
#include <string>

extern "C" LIB_API int init_classifier(const char* data_filename, const char* cfg_filename, const char* weight_filename);
extern "C" LIB_API int predict_top_classifier(const char* image_filename, classifier_t_container & result, int top = 0);
extern "C" LIB_API size_t get_class_name(int class_id, char* result);
extern "C" LIB_API int dispose_classifier();

class Classifier
{
public:
	
	LIB_API Classifier(std::string data_filename, std::string cfg_filename, std::string weight_filename);
	LIB_API std::vector<classifier_t> predict(std::string image_filename, int top = 0);
	LIB_API char* get_class_name(int class_id);
	LIB_API ~Classifier();

private:
	network net;
	std::string _cfg_filename, _weight_filename, _data_filename;
	char** names;
	int classes_count;
	int conf_top;
	int* indexes;
};


#endif // __cplusplus

#endif    // YOLO_CLASSIFIER_CLASS_HPP

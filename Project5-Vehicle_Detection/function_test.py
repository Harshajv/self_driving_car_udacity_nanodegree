from functions_detection import *
from functions_feat_extraction import *
import cv2
import pickle
from config import root_data_non_vehicle, root_data_vehicle, feat_extraction_params
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



if __name__ == '__main__':
    test_image=mpimg.imread("./test_images/test1.jpg")
    # load pretrained svm classifier
    svc = pickle.load(open('data/svm_trained.pickle', 'rb'))

    # load feature scaler fitted on training data
    feature_scaler = pickle.load(open('data/feature_scaler.pickle', 'rb'))

    # load parameters used to perform feature extraction
    feat_extraction_params = pickle.load(open('data/feat_extraction_params.pickle', 'rb'))


    h, w, c = test_image.shape
    draw_image = np.copy(test_image)
    test_image = test_image.astype(np.float32)/255
            
    windows = slide_window(test_image, x_start_stop=[None, None], y_start_stop=[h//2, None],xy_window=(64, 64), xy_overlap=(0.8, 0.8))
    print(windows)
    window_draw=draw_boxes(test_image,windows,color=(255,0,0))
    plt.imshow(window_draw)
    #hot_windows = search_windows(test_image, windows, svc, feature_scaler, feat_extraction_params)
    #window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    #plt.imshow(cv2.cvtColor(window_img, cv2.COLOR_BGR2RGB))
    plt.show()
    #cv2.imwrite("./output_images"+str(test1_output.jpg),window_img)
    print("done")






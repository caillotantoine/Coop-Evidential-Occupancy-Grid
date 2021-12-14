CXX=nvcc
CFLAGS = -I/usr/include/opencv4/opencv -I/usr/include/opencv4
LIBS = -lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_quality -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_ml -lopencv_videostab -lopencv_videoio -lopencv_viz -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core

target_1=rasterizer
target_2=merger

# gcc -shared -Wl,-soname,PixAvg -o PixAvg.so -fPIC fastAverage.c
# --compiler-options '-fPIC'

#  -fPIC -fdiagnostics-color=always -Wall
#https://forums.developer.nvidia.com/t/shared-library-creation/4776/2
#https://www.geeksforgeeks.org/how-to-call-c-c-from-python/


all: $(target_1).so $(target_1) $(target_2) $(target_2).so

$(target_1): $(target_1).o 
	$(CXX) -o $@ $(CFLAGS) $(LIBS) $?

$(target_1).so: $(target_1).o
	$(CXX) --compiler-options '-fPIC' -shared $(CFLAGS) $(LIBS) $? -o $@ 

$(target_1).o: $(target_1).cu
	$(CXX) -std=c++11 -c -g --compiler-options '-fPIC' $(CFLAGS) $(LIBS) $?

$(target_2): $(target_2).o 
	$(CXX) -o $@ $(CFLAGS) $(LIBS) $?

$(target_2).so: $(target_2).o
	$(CXX) --compiler-options '-fPIC' -shared $(CFLAGS) $(LIBS) $? -o $@ 

$(target_2).o: $(target_2).cu
	$(CXX) -std=c++11 -c -g --compiler-options '-fPIC' $(CFLAGS) $(LIBS) $?

clean:
	$(RM) $(target_1) $(target_2) *.o *.so


# all: test

# test:
# 	$(CC) $(target).cpp $(CFLAGS) $(LIBS) -o $(target) 

# lib:
# 	$(CC) $(CFLAGS) $(LIBS) -shared -Wl,-soname,$(target) -o $(target).so -fPIC $(target).cpp

# run: test
# 	./$(target)

# clean:
# 	rm -f $(target).so $(target)
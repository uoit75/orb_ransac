# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/liulang/for_nice-slam/orb_gauss_ransac_alter

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/liulang/for_nice-slam/orb_gauss_ransac_alter/build

# Include any dependencies generated for this target.
include CMakeFiles/orb.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/orb.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/orb.dir/flags.make

CMakeFiles/orb.dir/orb.cpp.o: CMakeFiles/orb.dir/flags.make
CMakeFiles/orb.dir/orb.cpp.o: ../orb.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liulang/for_nice-slam/orb_gauss_ransac_alter/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/orb.dir/orb.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orb.dir/orb.cpp.o -c /home/liulang/for_nice-slam/orb_gauss_ransac_alter/orb.cpp

CMakeFiles/orb.dir/orb.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orb.dir/orb.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liulang/for_nice-slam/orb_gauss_ransac_alter/orb.cpp > CMakeFiles/orb.dir/orb.cpp.i

CMakeFiles/orb.dir/orb.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orb.dir/orb.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liulang/for_nice-slam/orb_gauss_ransac_alter/orb.cpp -o CMakeFiles/orb.dir/orb.cpp.s

CMakeFiles/orb.dir/orb.cpp.o.requires:

.PHONY : CMakeFiles/orb.dir/orb.cpp.o.requires

CMakeFiles/orb.dir/orb.cpp.o.provides: CMakeFiles/orb.dir/orb.cpp.o.requires
	$(MAKE) -f CMakeFiles/orb.dir/build.make CMakeFiles/orb.dir/orb.cpp.o.provides.build
.PHONY : CMakeFiles/orb.dir/orb.cpp.o.provides

CMakeFiles/orb.dir/orb.cpp.o.provides.build: CMakeFiles/orb.dir/orb.cpp.o


CMakeFiles/orb.dir/ORBmatcher.cc.o: CMakeFiles/orb.dir/flags.make
CMakeFiles/orb.dir/ORBmatcher.cc.o: ../ORBmatcher.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liulang/for_nice-slam/orb_gauss_ransac_alter/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/orb.dir/ORBmatcher.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orb.dir/ORBmatcher.cc.o -c /home/liulang/for_nice-slam/orb_gauss_ransac_alter/ORBmatcher.cc

CMakeFiles/orb.dir/ORBmatcher.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orb.dir/ORBmatcher.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liulang/for_nice-slam/orb_gauss_ransac_alter/ORBmatcher.cc > CMakeFiles/orb.dir/ORBmatcher.cc.i

CMakeFiles/orb.dir/ORBmatcher.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orb.dir/ORBmatcher.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liulang/for_nice-slam/orb_gauss_ransac_alter/ORBmatcher.cc -o CMakeFiles/orb.dir/ORBmatcher.cc.s

CMakeFiles/orb.dir/ORBmatcher.cc.o.requires:

.PHONY : CMakeFiles/orb.dir/ORBmatcher.cc.o.requires

CMakeFiles/orb.dir/ORBmatcher.cc.o.provides: CMakeFiles/orb.dir/ORBmatcher.cc.o.requires
	$(MAKE) -f CMakeFiles/orb.dir/build.make CMakeFiles/orb.dir/ORBmatcher.cc.o.provides.build
.PHONY : CMakeFiles/orb.dir/ORBmatcher.cc.o.provides

CMakeFiles/orb.dir/ORBmatcher.cc.o.provides.build: CMakeFiles/orb.dir/ORBmatcher.cc.o


CMakeFiles/orb.dir/ORBextractor.cc.o: CMakeFiles/orb.dir/flags.make
CMakeFiles/orb.dir/ORBextractor.cc.o: ../ORBextractor.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liulang/for_nice-slam/orb_gauss_ransac_alter/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/orb.dir/ORBextractor.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orb.dir/ORBextractor.cc.o -c /home/liulang/for_nice-slam/orb_gauss_ransac_alter/ORBextractor.cc

CMakeFiles/orb.dir/ORBextractor.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orb.dir/ORBextractor.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liulang/for_nice-slam/orb_gauss_ransac_alter/ORBextractor.cc > CMakeFiles/orb.dir/ORBextractor.cc.i

CMakeFiles/orb.dir/ORBextractor.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orb.dir/ORBextractor.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liulang/for_nice-slam/orb_gauss_ransac_alter/ORBextractor.cc -o CMakeFiles/orb.dir/ORBextractor.cc.s

CMakeFiles/orb.dir/ORBextractor.cc.o.requires:

.PHONY : CMakeFiles/orb.dir/ORBextractor.cc.o.requires

CMakeFiles/orb.dir/ORBextractor.cc.o.provides: CMakeFiles/orb.dir/ORBextractor.cc.o.requires
	$(MAKE) -f CMakeFiles/orb.dir/build.make CMakeFiles/orb.dir/ORBextractor.cc.o.provides.build
.PHONY : CMakeFiles/orb.dir/ORBextractor.cc.o.provides

CMakeFiles/orb.dir/ORBextractor.cc.o.provides.build: CMakeFiles/orb.dir/ORBextractor.cc.o


CMakeFiles/orb.dir/Frame.cc.o: CMakeFiles/orb.dir/flags.make
CMakeFiles/orb.dir/Frame.cc.o: ../Frame.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/liulang/for_nice-slam/orb_gauss_ransac_alter/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/orb.dir/Frame.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/orb.dir/Frame.cc.o -c /home/liulang/for_nice-slam/orb_gauss_ransac_alter/Frame.cc

CMakeFiles/orb.dir/Frame.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/orb.dir/Frame.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/liulang/for_nice-slam/orb_gauss_ransac_alter/Frame.cc > CMakeFiles/orb.dir/Frame.cc.i

CMakeFiles/orb.dir/Frame.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/orb.dir/Frame.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/liulang/for_nice-slam/orb_gauss_ransac_alter/Frame.cc -o CMakeFiles/orb.dir/Frame.cc.s

CMakeFiles/orb.dir/Frame.cc.o.requires:

.PHONY : CMakeFiles/orb.dir/Frame.cc.o.requires

CMakeFiles/orb.dir/Frame.cc.o.provides: CMakeFiles/orb.dir/Frame.cc.o.requires
	$(MAKE) -f CMakeFiles/orb.dir/build.make CMakeFiles/orb.dir/Frame.cc.o.provides.build
.PHONY : CMakeFiles/orb.dir/Frame.cc.o.provides

CMakeFiles/orb.dir/Frame.cc.o.provides.build: CMakeFiles/orb.dir/Frame.cc.o


# Object files for target orb
orb_OBJECTS = \
"CMakeFiles/orb.dir/orb.cpp.o" \
"CMakeFiles/orb.dir/ORBmatcher.cc.o" \
"CMakeFiles/orb.dir/ORBextractor.cc.o" \
"CMakeFiles/orb.dir/Frame.cc.o"

# External object files for target orb
orb_EXTERNAL_OBJECTS =

orb: CMakeFiles/orb.dir/orb.cpp.o
orb: CMakeFiles/orb.dir/ORBmatcher.cc.o
orb: CMakeFiles/orb.dir/ORBextractor.cc.o
orb: CMakeFiles/orb.dir/Frame.cc.o
orb: CMakeFiles/orb.dir/build.make
orb: /usr/local/lib/libopencv_dnn.so.3.4.15
orb: /usr/local/lib/libopencv_highgui.so.3.4.15
orb: /usr/local/lib/libopencv_ml.so.3.4.15
orb: /usr/local/lib/libopencv_objdetect.so.3.4.15
orb: /usr/local/lib/libopencv_shape.so.3.4.15
orb: /usr/local/lib/libopencv_stitching.so.3.4.15
orb: /usr/local/lib/libopencv_superres.so.3.4.15
orb: /usr/local/lib/libopencv_videostab.so.3.4.15
orb: /usr/local/lib/libopencv_viz.so.3.4.15
orb: /usr/local/lib/libopencv_calib3d.so.3.4.15
orb: /usr/local/lib/libopencv_features2d.so.3.4.15
orb: /usr/local/lib/libopencv_flann.so.3.4.15
orb: /usr/local/lib/libopencv_photo.so.3.4.15
orb: /usr/local/lib/libopencv_video.so.3.4.15
orb: /usr/local/lib/libopencv_videoio.so.3.4.15
orb: /usr/local/lib/libopencv_imgcodecs.so.3.4.15
orb: /usr/local/lib/libopencv_imgproc.so.3.4.15
orb: /usr/local/lib/libopencv_core.so.3.4.15
orb: CMakeFiles/orb.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/liulang/for_nice-slam/orb_gauss_ransac_alter/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable orb"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/orb.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/orb.dir/build: orb

.PHONY : CMakeFiles/orb.dir/build

CMakeFiles/orb.dir/requires: CMakeFiles/orb.dir/orb.cpp.o.requires
CMakeFiles/orb.dir/requires: CMakeFiles/orb.dir/ORBmatcher.cc.o.requires
CMakeFiles/orb.dir/requires: CMakeFiles/orb.dir/ORBextractor.cc.o.requires
CMakeFiles/orb.dir/requires: CMakeFiles/orb.dir/Frame.cc.o.requires

.PHONY : CMakeFiles/orb.dir/requires

CMakeFiles/orb.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/orb.dir/cmake_clean.cmake
.PHONY : CMakeFiles/orb.dir/clean

CMakeFiles/orb.dir/depend:
	cd /home/liulang/for_nice-slam/orb_gauss_ransac_alter/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/liulang/for_nice-slam/orb_gauss_ransac_alter /home/liulang/for_nice-slam/orb_gauss_ransac_alter /home/liulang/for_nice-slam/orb_gauss_ransac_alter/build /home/liulang/for_nice-slam/orb_gauss_ransac_alter/build /home/liulang/for_nice-slam/orb_gauss_ransac_alter/build/CMakeFiles/orb.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/orb.dir/depend


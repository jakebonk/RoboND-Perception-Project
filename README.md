## Project: Perception Pick & Place
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Complete Exercise 1 steps. Pipeline for filtering and RANSAC plane fitting implemented.
To correctly complete exercise 1 we need to fill out the pcl_callback() function. You can find the whole RANSAC.py file in the code folder. We start with the cloud data in the variable `cloud`. This file variable contains all the data captured from our RGBD camera. We need to set the leaf size of the data. The leaf size is the size of each point in the captured.

`vox = cloud.make_voxel_grid_filter()
LEAF_SIZE = 0.01
vox.set_leaf_size(LEAF_SIZE,LEAF_SIZE,LEAF_SIZE)
cloud = vox.filter()`

We start by creating a voxel grid filter and set the leaf size to 0.01. We then end the code segment by applying the filter.

Next we create and apply a passthrough filter which is very similar to how cropping works in 2D.

`passthrough = cloud.make_passthrough_filter()
filter_axis = 'z'
passthrough.set_filter_field_name(filter_axis)
axis_min = 0.6
axis_max = 1.1
passthrough.set_filter_limits(axis_min,axis_max)
cloud = passthrough.filter()`

We set the z-axis on line 2 as the axis we want apply a passthrough filter on. With this we take the cloud data points that are between 0.6 and 1.1 on the z-axis. We end this again with applying the filter back to our cloud variable.

Next we apply the RANSAC plane segmentation.

`seg = cloud.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)`

RANSAC plane segmentation already exist in the PCL library so are main focus will be correctly setting the max_distance variable.

`max_distance = 0.01
seg.set_distance_threshold(max_distance)
inliers, coefficients = seg.segment()
extracted_inliers = cloud_filtered.extract(inliers, negative=True)`

This code segment will set the max_distance to 0.01 and allow us to keep the points that fit the model.

This is the result of segmenting the points.
![Extracted Inliers](https://github.com/jakebonk/RoboND-Perception-Project/blob/master/images/extracted_inliers.png?raw=true)

We end this function by cleaning up any outliers that produce noise in the background.

#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

In this exercise we focus on clustering the filtered results. The code for exercise 2 is stored in the template.py file in the code folder. Much of the code in this section is reused from Exercise 1 with slightly changed variables, so I'll only explain the clustering that was implemented. In class we learned about the differences between DBSCAN and K-means. For our application DBSCAN will work better because we want to search for a dynamic about of objects and group them.

In the code we set the cluster tolerance to 0.04 which is how far a point can be to be considered a neighbor. Next we set the minimum and maximum size of a cluster. I found the size 150 - 1500 to work well.

Next we implement DBSCAN into the code and publish the cloud data to rospy. This will let us view our clusters in RVIZ.

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.

I increased the training from 5 to 20 and noticed that the accuracy increased with it.
![Confusion Matrix](https://github.com/jakebonk/RoboND-Perception-Project/blob/master/images/confusion_matrix.png?raw=true)


### Pick and Place Setup

Before running the tests for the object recognition I trained my model on the objects in 100 different poses to increase the recognition of the objects.
![Confusion Matrix Project](https://github.com/jakebonk/RoboND-Perception-Project/blob/master/images/confusion_matrix_project.png?raw=true)

Looking the first output_1.yaml file we see that have correctly detected 3/3 of the objects in the scene
![World 1](https://github.com/jakebonk/RoboND-Perception-Project/blob/master/images/world1.png?raw=true)

Next in output_2.yaml we successfully identify 4/5 objects correctly. We misidentify book as soap.
![World 2](https://github.com/jakebonk/RoboND-Perception-Project/blob/master/images/world2.png?raw=true)

Next in output_3.yaml we successfully identify 7/8 objects correctly. We don't even identify the glue bottle.
![World 3](https://github.com/jakebonk/RoboND-Perception-Project/blob/master/images/world3.png?raw=true)

One thing that I modified with my object recognition was to have it rerun the recognition after it each pick and place action. On the third test world I noticed that it did not find the glue model initially. After several of the objects were removed it recognized the glue but after it had already skipped it in the search. I tried lowering the minimum size for clusters thinking that it may have not recognized it hiding behind something but it resulted in misidentifying other objects. I believe that to increase accuracy that we can train the model on objects that are missing sections or are being blocked.
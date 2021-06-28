## Face Recognition with Liveliness Detection

This repository is a facial recognition program with integrated liveliness detection. Its based around the dlib-based [face_recognition](https://pypi.org/project/face-recognition/) Python library. 

Environment used:
- Windows 10
- Python 3.8.6
- CUDA 11.2.0
- cuDNN 8.1.0.77

### Theory

The basic premise I have used for the liveliness detection is that for a person to be real, the person will have to blink (obviously). The program runs for 7 seconds, which is observed to be the most the average human waits between blinks.

Unlike other liveliness detection algorithms that I have tried out, the algorithm I have used is almost unaffected by the distance of the person from the camera, as long as their face is clearly visible and detectable by the camera.

Other algorithms use something called as a **blink-ratio** which is basically the ratio between the height and the width of the eyes (average of both eyes).If this ratio dips below a specific value, it counts as a blink. This method does work, but it works only when the user is static and at a specific distance from the camera.

>Single blink graph:
![](https://github.com/AnveshakR/facerecog/blob/master/images/single%20blink.jpg?raw=true)

I have taken a slightly different approach, in that when the user blinks, the height of the detected eye decreases suddenly. But this change is sometimes imperceptible at certain ranges from the camera, or sometimes due to the user being at a distance from the camera such that the defined threshold fails. My approach consists of taking the **slopes** of all the adjacent points. This introduces a bit of linearity as shown by this graph.

![](https://github.com/AnveshakR/facerecog/blob/master/images/far_to_close_4_steps.png?raw=true)

For the above graph, I was moving towards the camera, stopping at 4 intervals and blinking 3 distinct times for each interval.
The *orange* graph shows the raw graph between the eye widths captured in by the program. As we can see, defining a threshold for any one distance will cause the other ones to fail.
The *blue* graph plots the slopes between each adjacent pair of points. In this we can clearly see the blinks as sudden changes in the graph. Thus we can very clearly define thresholds for measured distances from the camera. 

To define distances from the camera, I have utilised the lengths of the eye of the user as a constant parameter. This is because, for a particular distance, the width of the eye is irrespective of the user blinking or not. I have divided the distance from the camera into 4 regions.

>Slope vs Width
![](https://github.com/AnveshakR/facerecog/blob/master/images/width_slope.png?raw=true)

As we can see, the widths are almost constant for each distance. From this graph, we can define 4 regions, each with a different threshold for the corresponding slope values. Instead of hardcoding these values in the code, I have included these values in the .env file. The values I have taken are:

- width <= 23, slope <= -0.75
- width > 23, width <= 38, slope <= -2.75
- width > 38, width <= 54, slope <= -4.75
- width > 54, slope <= -5.75

The face recognition is based on the faces saved in a separate folder, whose absolute path is defined in the .env file too. The faces should be labelled with the appropriate name by which they should be recognized, like Anveshak.jpg. 

### The .env file
The .env file I am using looks like this. You can copy it for your own reference.

	faces_directory = <absolute path of the faces folder>
	width1 = 23
	slope1 = -0.75

	width2 = 38
	slope2 = -2.75

	width3 = 54
	slope3 = -4.75

	slope4 = -5.75

### Folder Layout
The folder layout looks something like this:

	│   .env
	│   .gitignore
	│   final_facerecog.py
	│   requirements.txt
	│
	├───images
	│       	far_to_close_4_steps.png
	│       	single blink.jpg
	│       	width_slope.png
	│
	└───individual programs
				.env
				face_recogniton.py
				liveliness_detector.py

### Future Checklist
- [ ] Proper commenting lol
- [ ] Adding accomodation for multiple faces
- [ ] GUI
- [ ] Auto/manual setup feature
- [ ] Adding new faces via the program instead of manually

### Thanks!
Please feel free to open issues or ask me directly if you have a doubt!
# ChoreoAssist
"ChoreoAssist" was created during the course "Designing Intelligence in Interaction" during my second year of Masters in Industrial Design at the TU/e with the help of other three teammates. Intelligent systems are becoming increasingly important in the area of sports and can support athletes in all kinds of disciplines. ChoreoAssist is developed to support dance teachers and choreographers in their work. To prevent them from being distracted when developing their dances, the software helps them document their progress.

ChoreoAssist uses a Kinect device to be able to detect the dancers poses and identify their names. The detection is done by analyzing joint coordinates and classifying them with supervised learning. Four classification algorithms were used in training and testing the system: Neural Networks (Multilayered Perceptron), Decision Trees (Random Forest and Gradient Tree Boost) and Support Vector Machines. Being the only technical group member, my responsbilities were  coding, gathering the data, and figuring out why the first data set failed to be a good one, and trying different software to train and test the algorithms. 
 
Kinect
------
The detection and data-collection software behind ChoreoAssist is programmed in Processing, which has one library for Kinect V2 called KinectPV2. The library provides built-in methods for body detection and allows for obtaining information from the color camera and the infrared camera such as the key joints of the body (24 joints), bones which connect the joints, and the state of the hands (open or close). Our aim was to analyze six ballet positions which are determined by certain positions of the limbs. The used algorithm treats joints and bones as simple points and segments in the 3D space, making it simple to extract the position and orientation of the dancers’ limbs. 

Machine Learning 
---------------
Four classification algorithms were used in training and testing the system: Neural Networks (Multilayered Perceptron), Decision Trees (Random Forest and Gradient Tree Boost) and Support Vector Machines. I made use of Neuroph (http://neuroph.sourceforge.net/) and trained 4 different multilayered perceptrons (MLPs). Then,  I made use of the SMILE library (https://haifengl.github.io/) v1.4. and the Eclipse IDE to implement a MLP which uses backpropagation. Two training sets were used: the combined sets from all participants including the negatives and excluding the negatives. The training set had roughly 2700 lines (including negatives), 2000 (excluding negatives) while the testing set had 285 lines (including negatives) and 150 (excluding negatives) for each body part (upper/lower). The number of negatives was so high because there are much more possible variations of
angles of “no position” rather than variations of angles for the three moves. 

If you wish to learn more about this project, visit: https://www.irinabiancaserban.com/choreo-assist

# Photo Wake-Up: 3D Character Animation from a Single Photo

In recent years, tremendous amount of progress is being made in the field of 3D Machine Learning, which is an interdisciplinary field that fuses computer vision, computer graphics and machine learning. Photo wake-up belongs to that field. Researchers at the University of Washington have developed this technique for animating a human subject (such as walking toward the screen, running, sitting, and jumping) from a single photo. The technique is demonstrated by using a variety of singleimage inputs, including photographs, realistic illustrations, cartoon drawings, and abstracted human forms. The output animation can be played as a video, viewed interactively on a monitor, and as an augmented or virtual reality experience, where a user with headset can enjoy the central figure of a photo coming out into the real world.

![Alt text](https://github.com/lev1khachatryan/Photo_Wake-Up/blob/master/doc/_assets/main_result.png)
**Figure 1**: Given a single photo as input (far left), model creates a 3D animatable version of the subject, which can now walk towards the viewer (middle). The 3D result can be experienced in augmented reality (right); in the result above the user has virtually hung the artwork with a HoloLens headset and can watch the character run out of the painting from different views.


The overall system works as follows (Fig. 2): First apply state-of-the-art algorithms to perform person detection, segmentation, and 2D pose estimation. From the results, devise a method to construct a rigged mesh. Any 3D motion sequence can then be used to animate the rigged mesh. More specifically, Mask R-CNN is used for person detection and segmentation. 2D body pose is estimated using Convolutional pose machines, and person segmentation is refined using Dense CRF. Once the person is segmented out of the photo, PatchMatch (a randomized correspondence algorithm for structural image editing) is applied to fill in the regions where the person used to be.

![Alt text](https://github.com/lev1khachatryan/Photo_Wake-Up/blob/master/doc/_assets/architecture.png)
**Figure 2**: Overview of our method. Given a photo, person detection, 2D pose estimation, and person segmentation, is performed using off-the-shelf algorithms. Then, A SMPL template model is fit to the 2D pose and projected into the image as a normal map and a skinning map. The core of our system is: find a mapping between person’s silhouette and the SMPL silhouette, warp the SMPL normal/skinning maps to the output, and build a depth map by integrating the warped normal map. This process is repeated to simulate the model’s back view and combine depth and skinning maps to create a complete, rigged 3D mesh. The mesh is further textured, and animated using motion capture sequences on an unpainted background.

## An Overview on Techniques for Photo Wake-Up

***Mask R-CNN***: Used for person detection and segmentation which is based on Faster R-CNN, so let’s begin by briefly reviewing this detector. Faster R-CNN consists of two stages. The first stage, called a Region Proposal Network (RPN), proposes candidate object bounding boxes (we also call them region of interest). The second stage, which is in essence Fast R-CNN, extracts features using RoIPool from each candidate box and performs classification and bounding-box regression. The features used by both stages can be shared for faster inference.

Mask R-CNN adopts the same two-stage procedure, with an identical first stage (which is RPN). In the second stage, in parallel to predicting the class and box offset, Mask R-CNN also outputs a binary mask for each RoI (region of interest).

![Alt text](https://github.com/lev1khachatryan/Photo_Wake-Up/blob/master/doc/_assets/mask%20r-cnn.png)
**Figure 3*. The Mask R-CNN. Key element is RoIAlign which is the main missing piece of Fast/Faster R-CNN.

***Dense CRF***: As we mentioned above, person segmentation is refined using dense CRF. A common approach of pixel-level models for segmentation/detection (as Mask R-CNN does) is to pose problem as maximum a posteriori (MAP) inference in a conditional random field (CRF) defined over pixels or image patches. The CRF potentials incorporate smoothness terms that maximize label agreement between similar pixels, and can integrate more elaborate terms that model contextual relationships between object classes.

Basic CRF models are composed of unary potentials on individual pixels or image patches and pairwise potentials on neighboring pixels or patches but fully connected (also called dense) CRF establishes pairwise potentials on all pairs of pixels in the image. The pairwise edge potentials are defined by a linear combination of Gaussian kernels in an arbitrary feature space. The algorithm is based on a mean field approximation to the CRF distribution.

![Alt text](https://github.com/lev1khachatryan/Photo_Wake-Up/blob/master/doc/_assets/crf.png)
**Figure 4**: Pixel-level classification with a fully connected CRF. (a) Input image from the MSRC-21 dataset. (b) The response of unary classifiers. (c) Classification produced by the Robust 𝑃𝑛 CRF. (d) Classification produced by MCMC inference in a fully connected pixel-level CRF model; the algorithm was run for 36 hours and only partially converged for the bottom image. (e) Classification produced by our inference algorithm in the fully connected model in 0.2 seconds.

***PatchMatch***: Used to fill in the holes in the person after being segmented. The core of the system is the algorithm for computing patch correspondences. We define a nearest-neighbor field (NNF) as a function <img src="https://render.githubusercontent.com/render/math?math=f : A \to R^{2}">  of offsets, defined over all possible patch coordinates (locations of patch centers) in image 𝐴, for some distance function of two patches 𝐷. Given patch coordinate 𝑎 in image 𝐴 and its corresponding nearest neighbor 𝑏 in image 𝐵, 𝑓(𝑎) is simply 𝑏 − 𝑎 . We refer to the values of 𝑓 as offsets, and they are stored in an array whose dimensions are those of A.












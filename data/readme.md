

\# BCCL Dataset



This directory provides representative detection videos and annotated evaluation data associated with the paper \*\*"Edge-Cloud Collaborative Continual Learning in Smart Livestock Monitoring"\*\*.



The data were collected from a commercial pig farm located in Luoding County, Yunfu City, Guangdong Province, China. The pigs were introduced to the farm around April 10, 2025, and video data were collected from June 21 to September 10, 2025.



All videos were recorded using an overhead monitoring system, providing \*\*top-view observations\*\* of the pigs. Each video frame contains approximately \*\*40–50 pigs\*\*. The dataset covers three pig growth stages and four representative monitoring conditions within each stage.



\## 1. Data Organization



The released dataset contains two types of data:



\- \*\*Detection videos:\*\* Representative video sequences used for continual learning experiments.

\- \*\*Evaluation data:\*\* Images with manually annotated bounding boxes and pig behavior labels for offline evaluation.



The recommended directory structure is:



```text

data/

├── README.md

├── videos/

│   └── ...                  # Detection video sequences

└── evaluation/

&#x20;   ├── images/              # Evaluation images

&#x20;   └── labels/              # Bounding-box and behavior annotations

```



Only a portion of the original dataset is provided in this repository.



\## 2. Detection Video Data



The detection video data were used to conduct long-term continual learning experiments.



A total of \*\*36 video sequences\*\* were selected, covering three growth stages (T1, T2, and T3) and four representative monitoring conditions (Day, Night, Feeding, and Light). For each growth stage, three video sequences were selected under each monitoring condition.



\### 2.1 Pig Growth Stages



The monitoring data were divided into three growth stages according to changes in pig body size, crowding, and occlusion.



| Stage | Collection Period | Approximate Age | Characteristics |

|---|---|---|---|

| T1 | June 21 – July 10, 2025 | 71–90 days | Smaller body sizes, relatively low crowding, and limited mutual occlusion. |

| T2 | July 11 – August 10, 2025 | 91–120 days | Moderate body sizes, increased crowding, and partial occlusion. |

| T3 | August 11 – September 10, 2025 | 121–150 days | Larger body sizes, greater crowding, and more severe occlusion. |



\### 2.2 Monitoring Conditions



Each growth stage includes four representative monitoring conditions:



| Condition | Sequence Duration | Description |

|---|---|---|

| Day | 3 hours | Daytime monitoring with relatively clear illumination. |

| Night | 3 hours | Nighttime monitoring under low-light conditions. |

| Feeding | 1 hour | Feeding periods when pigs gather around the feeding trough, resulting in increased crowding and occlusion. |

| Light | 0.5 hours | Monitoring under changing or uneven artificial illumination, which may introduce localized shadows. |



The feeding trough is located near the lower center and lower sides of the camera view. Pig aggregation around the trough during feeding and uneven artificial lighting introduce additional challenges for detection.



\## 3. Annotated Evaluation Data



A separate offline evaluation dataset was constructed to assess pig detection and behavior recognition performance.



The complete evaluation dataset contains \*\*2,400 images\*\*, sampled from additional video segments independent of the 36 video sequences used for continual learning.



The evaluation dataset covers the same three pig growth stages and four monitoring conditions, resulting in 12 stage–condition combinations.



All evaluation images were manually annotated with pig bounding boxes and three behavior categories:



\- `0`: stand

\- `1`: lie

\- `2`: eat



The released evaluation samples contain corresponding images and annotation files. The annotation files provide the bounding boxes and behavior labels of pigs in each image.



\## 4. Behavior Classes and Annotation Criteria



Pig behaviors are classified into three categories based on observable body posture, head position, and spatial relationships with the feeding trough in top-view images.



\### Class 0: stand



The pig exhibits an upright standing posture, with its limbs visibly supporting the body.



This category also includes partially raised postures in which the forelimbs support the front of the body and the head is lifted.



\### Class 1: lie



The pig rests with its body lowered toward or against the floor, typically with the trunk and forequarters in a lying posture.



This category also includes postures in which the hind limbs partially support the rear of the body, while the forequarters remain against the floor and the head stays close to the ground.



\### Class 2: eat



The pig is positioned near the feeding trough and exhibits an observable feeding posture, with its head lowered from above and extended toward or into the trough.



In the top-view images, the feeding troughs are located along the lower central area and the lower left and right sides of the frame.



\### Annotation Notes



\- The behavior labels are determined from visible pig postures in top-view images.

\- Proximity to the feeding trough alone is insufficient for classification as `eat`; the pig must also exhibit a visible feeding posture.

\- A pig with a standing body posture but a clearly lowered head directed into the feeding trough is classified as `eat`.

\- The `eat` category represents an observable feeding posture rather than a direct measurement of food consumption.



The class IDs are consistently defined as:



| Class ID | Behavior |

|---|---|

| 0 | stand |

| 1 | lie |

| 2 | eat |



\## 5. Data Availability



Due to the large size of the original dataset, only a portion of the detection videos and annotated evaluation data is uploaded to this repository.



For access to the complete dataset, please contact the authors at yezhenhua@stu.scau.edu.cn. Data sharing is subject to applicable permissions and restrictions associated with the commercial farm recordings.



\## 6. Citation



If you use the released data in your research, please cite the associated paper:



\*\*Edge-Cloud Collaborative Continual Learning in Smart Livestock Monitoring\*\*



Full bibliographic information will be added after publication.



\## 7. Usage and Redistribution



Please comply with any access and redistribution restrictions associated with the farm recordings and annotations.



The repository's software license does not automatically grant unrestricted rights to the dataset.




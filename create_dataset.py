import cv2

# import xml.etree.ElementTree as ET
import scipy.io
import os

from chainercv.utils import read_image

person_1 = 0
person_2 = 0
person_3 = 0
person_4 = 0
person_5 = 0
person_6 = 0
person_7 = 0
person_8 = 0

count = 0
annotation = scipy.io.loadmat(
    r"C:\Users\andre\Desktop\foto and labels\wider_face_split\wider_face_split\wider_face_train.mat"
)

for i in range(len(annotation["event_list"])):
    event = annotation["event_list"][i, 0][0]
    print(event)

    # append mode
    # if event == "0--Parade":
    if count < 2000:
        # if event in folder_to_consider:

        for j in range(len(annotation["file_list"][i, 0])):
            file = annotation["file_list"][i, 0][j, 0][0]
            filename = "{}.jpg".format(file)
            bboxs = annotation["face_bbx_list"][i, 0][j, 0]

            # if filename == "0_Parade_marchingband_1_45.jpg":
            if 1 == 1:

                swapped_bbxs = bboxs[:, [1, 0, 3, 2]]  #  (y,x,h,w)
                swapped_bbxs[:, 2:4] = swapped_bbxs[:, 2:4] + swapped_bbxs[:, 0:2]
                reading_path = (
                    "C:\\Users\\andre\\Desktop\\foto and labels\\WIDER_train\\WIDER_train\\images\\"
                    + event
                    + "\\"
                    + filename
                )
                writing_path = (
                    "C:\\Users\\andre\\Desktop\\wider_subset_images\\images\\"
                )
                file_to_be_write = (
                    "C:\\Users\\andre\\Desktop\\wider_subset_images\\images\\"
                    + filename
                )

                # Check whether the specified path exists or not

                if not os.path.exists(writing_path):

                    # Create a new directory because it does not exist
                    os.makedirs(writing_path)
                #    print("The new directory is created!")

                if (
                    os.path.exists(
                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                        + "\\"
                        + str(count)
                        + ".txt"
                    )
                    == False
                ):
                    file1 = open(
                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                        + "\\"
                        + str(count)
                        + ".txt",
                        "w",
                    )
                    file1.close()

                file1 = open(
                    r"C:\Users\andre\Desktop\wider_subset_images\annots"
                    + "\\"
                    + str(count)
                    + ".txt",
                    "a",
                )

                image = cv2.imread(reading_path)
                image_only_writing = cv2.imread(reading_path)

                image_area = (
                    image_only_writing.shape[0] * image_only_writing.shape[1]
                )

                if not (
                    swapped_bbxs[0][0] == 0
                    and swapped_bbxs[0][1] == 0
                    and swapped_bbxs[0][2] == 0
                    and swapped_bbxs[0][3] == 0
                ):

                    if len(swapped_bbxs) == 1:  # and person_1 < 125:
                        ok = 1
                        for k in range(len(swapped_bbxs)):

                            annotation_area = (
                                int(swapped_bbxs[k][3]) - int(swapped_bbxs[k][1])
                            ) * (int(swapped_bbxs[k][2]) - int(swapped_bbxs[k][0]))

                            perc = (annotation_area / image_area) * 100

                            if perc < 8:
                                if (
                                    os.path.exists(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    == True
                                ):
                                    file1.close()
                                    os.remove(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    ok = 0
                                    break
                            else:
                                cv2.rectangle(
                                    image,
                                    (swapped_bbxs[k][1], swapped_bbxs[k][0]),
                                    (swapped_bbxs[k][3], swapped_bbxs[k][2]),
                                    (0, 255, 0),
                                    2,
                                )

                                file1.write(
                                    str(swapped_bbxs[k][1])
                                    + " "
                                    + str(swapped_bbxs[k][0])
                                    + " "
                                    + str(swapped_bbxs[k][3])
                                    + " "
                                    + str(swapped_bbxs[k][2])
                                    + "\n"
                                )

                        if ok == 1:
                            cv2.imwrite(
                                writing_path + str(count) + ".jpg",
                                image_only_writing,
                            )
                            print(filename)
                            print(
                                "writing_path:" + writing_path + str(count) + ".jpg"
                            )
                            count += 1
                            person_1 += 1
                            file1.close()

                    elif len(swapped_bbxs) == 2:
                        ok = 1
                        for k in range(len(swapped_bbxs)):

                            annotation_area = (
                                int(swapped_bbxs[k][3]) - int(swapped_bbxs[k][1])
                            ) * (int(swapped_bbxs[k][2]) - int(swapped_bbxs[k][0]))

                            perc = (annotation_area / image_area) * 100

                            if perc < 5:
                                if (
                                    os.path.exists(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    == True
                                ):
                                    file1.close()
                                    os.remove(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    ok = 0
									
                                    break
                            else:
                                cv2.rectangle(
                                    image,
                                    (swapped_bbxs[k][1], swapped_bbxs[k][0]),
                                    (swapped_bbxs[k][3], swapped_bbxs[k][2]),
                                    (0, 255, 0),
                                    2,
                                )

                                file1.write(
                                    str(swapped_bbxs[k][1])
                                    + " "
                                    + str(swapped_bbxs[k][0])
                                    + " "
                                    + str(swapped_bbxs[k][3])
                                    + " "
                                    + str(swapped_bbxs[k][2])
                                    + "\n"
                                )

                        if ok == 1:
                            cv2.imwrite(
                                writing_path + str(count) + ".jpg",
                                image_only_writing,
                            )
                            print(filename)
                            print(
                                "writing_path:" + writing_path + str(count) + ".jpg"
                            )
                            count += 1
                            person_2 += 1
                            file1.close()

                    elif len(swapped_bbxs) == 3:
                        ok = 1
                        for k in range(len(swapped_bbxs)):

                            annotation_area = (
                                int(swapped_bbxs[k][3]) - int(swapped_bbxs[k][1])
                            ) * (int(swapped_bbxs[k][2]) - int(swapped_bbxs[k][0]))

                            perc = (annotation_area / image_area) * 100

                            if perc < 2:
                                if (
                                    os.path.exists(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    == True
                                ):
                                    file1.close()
                                    os.remove(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    ok = 0
                                    break
                            else:
                                cv2.rectangle(
                                    image,
                                    (swapped_bbxs[k][1], swapped_bbxs[k][0]),
                                    (swapped_bbxs[k][3], swapped_bbxs[k][2]),
                                    (0, 255, 0),
                                    2,
                                )

                                file1.write(
                                    str(swapped_bbxs[k][1])
                                    + " "
                                    + str(swapped_bbxs[k][0])
                                    + " "
                                    + str(swapped_bbxs[k][3])
                                    + " "
                                    + str(swapped_bbxs[k][2])
                                    + "\n"
                                )

                        if ok == 1:
                            cv2.imwrite(
                                writing_path + str(count) + ".jpg",
                                image_only_writing,
                            )
                            print(filename)
                            print(
                                "writing_path:" + writing_path + str(count) + ".jpg"
                            )
                            count += 1
                            person_3 += 1
                            file1.close()

                    elif len(swapped_bbxs) == 4:
                        ok = 1
                        for k in range(len(swapped_bbxs)):

                            annotation_area = (
                                int(swapped_bbxs[k][3]) - int(swapped_bbxs[k][1])
                            ) * (int(swapped_bbxs[k][2]) - int(swapped_bbxs[k][0]))

                            perc = (annotation_area / image_area) * 100

                            if perc < 1:
                                if (
                                    os.path.exists(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    == True
                                ):
                                    file1.close()
                                    os.remove(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    ok = 0
                                    break
                            else:
                                cv2.rectangle(
                                    image,
                                    (swapped_bbxs[k][1], swapped_bbxs[k][0]),
                                    (swapped_bbxs[k][3], swapped_bbxs[k][2]),
                                    (0, 255, 0),
                                    2,
                                )

                                file1.write(
                                    str(swapped_bbxs[k][1])
                                    + " "
                                    + str(swapped_bbxs[k][0])
                                    + " "
                                    + str(swapped_bbxs[k][3])
                                    + " "
                                    + str(swapped_bbxs[k][2])
                                    + "\n"
                                )

                        if ok == 1:
                            cv2.imwrite(
                                writing_path + str(count) + ".jpg",
                                image_only_writing,
                            )
                            print(filename)
                            print(
                                "writing_path:" + writing_path + str(count) + ".jpg"
                            )
                            count += 1
                            person_4 += 1
                            file1.close()

                    elif len(swapped_bbxs) == 5:
                        ok = 1
                        for k in range(len(swapped_bbxs)):

                            annotation_area = (
                                int(swapped_bbxs[k][3]) - int(swapped_bbxs[k][1])
                            ) * (int(swapped_bbxs[k][2]) - int(swapped_bbxs[k][0]))

                            perc = (annotation_area / image_area) * 100

                            if 1 != 1:
                                if (
                                    os.path.exists(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    == True
                                ):
                                    file1.close()
                                    os.remove(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    ok = 0
                                    break
                            else:
                                cv2.rectangle(
                                    image,
                                    (swapped_bbxs[k][1], swapped_bbxs[k][0]),
                                    (swapped_bbxs[k][3], swapped_bbxs[k][2]),
                                    (0, 255, 0),
                                    2,
                                )

                                file1.write(
                                    str(swapped_bbxs[k][1])
                                    + " "
                                    + str(swapped_bbxs[k][0])
                                    + " "
                                    + str(swapped_bbxs[k][3])
                                    + " "
                                    + str(swapped_bbxs[k][2])
                                    + "\n"
                                )

                        if ok == 1:
                            cv2.imwrite(
                                writing_path + str(count) + ".jpg",
                                image_only_writing,
                            )
                            print(filename)
                            print(
                                "writing_path:" + writing_path + str(count) + ".jpg"
                            )
                            count += 1
                            person_5 += 1
                            file1.close()

                    elif len(swapped_bbxs) == 6:
                        ok = 1
                        for k in range(len(swapped_bbxs)):

                            annotation_area = (
                                int(swapped_bbxs[k][3]) - int(swapped_bbxs[k][1])
                            ) * (int(swapped_bbxs[k][2]) - int(swapped_bbxs[k][0]))

                            perc = (annotation_area / image_area) * 100

                            if 1 != 1:
                                if (
                                    os.path.exists(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    == True
                                ):
                                    file1.close()
                                    os.remove(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    ok = 0
                                    break
                            else:
                                cv2.rectangle(
                                    image,
                                    (swapped_bbxs[k][1], swapped_bbxs[k][0]),
                                    (swapped_bbxs[k][3], swapped_bbxs[k][2]),
                                    (0, 255, 0),
                                    2,
                                )

                                file1.write(
                                    str(swapped_bbxs[k][1])
                                    + " "
                                    + str(swapped_bbxs[k][0])
                                    + " "
                                    + str(swapped_bbxs[k][3])
                                    + " "
                                    + str(swapped_bbxs[k][2])
                                    + "\n"
                                )

                        if ok == 1:
                            cv2.imwrite(
                                writing_path + str(count) + ".jpg",
                                image_only_writing,
                            )
                            print(filename)
                            print(
                                "writing_path:" + writing_path + str(count) + ".jpg"
                            )
                            count += 1
                            person_6 += 1
                            file1.close()

                    elif len(swapped_bbxs) == 7:
                        ok = 1
                        for k in range(len(swapped_bbxs)):

                            annotation_area = (
                                int(swapped_bbxs[k][3]) - int(swapped_bbxs[k][1])
                            ) * (int(swapped_bbxs[k][2]) - int(swapped_bbxs[k][0]))

                            perc = (annotation_area / image_area) * 100

                            if 1 != 1:
                                if (
                                    os.path.exists(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    == True
                                ):
                                    file1.close()
                                    os.remove(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    ok = 0
                                    break
                            else:
                                cv2.rectangle(
                                    image,
                                    (swapped_bbxs[k][1], swapped_bbxs[k][0]),
                                    (swapped_bbxs[k][3], swapped_bbxs[k][2]),
                                    (0, 255, 0),
                                    2,
                                )

                                file1.write(
                                    str(swapped_bbxs[k][1])
                                    + " "
                                    + str(swapped_bbxs[k][0])
                                    + " "
                                    + str(swapped_bbxs[k][3])
                                    + " "
                                    + str(swapped_bbxs[k][2])
                                    + "\n"
                                )

                        if ok == 1:
                            cv2.imwrite(
                                writing_path + str(count) + ".jpg",
                                image_only_writing,
                            )
                            print(filename)
                            print(
                                "writing_path:" + writing_path + str(count) + ".jpg"
                            )
                            count += 1
                            person_7 += 1
                            file1.close()

                    elif len(swapped_bbxs) == 8:
                        ok = 1
                        for k in range(len(swapped_bbxs)):

                            annotation_area = (
                                int(swapped_bbxs[k][3]) - int(swapped_bbxs[k][1])
                            ) * (int(swapped_bbxs[k][2]) - int(swapped_bbxs[k][0]))

                            perc = (annotation_area / image_area) * 100

                            if 1 != 1:
                                if (
                                    os.path.exists(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    == True
                                ):
                                    file1.close()
                                    os.remove(
                                        r"C:\Users\andre\Desktop\wider_subset_images\annots"
                                        + "\\"
                                        + str(count)
                                        + ".txt"
                                    )
                                    ok = 0
                                    break
                            else:
                                cv2.rectangle(
                                    image,
                                    (swapped_bbxs[k][1], swapped_bbxs[k][0]),
                                    (swapped_bbxs[k][3], swapped_bbxs[k][2]),
                                    (0, 255, 0),
                                    2,
                                )

                                file1.write(
                                    str(swapped_bbxs[k][1])
                                    + " "
                                    + str(swapped_bbxs[k][0])
                                    + " "
                                    + str(swapped_bbxs[k][3])
                                    + " "
                                    + str(swapped_bbxs[k][2])
                                    + "\n"
                                )

                        if ok == 1:
                            cv2.imwrite(
                                writing_path + str(count) + ".jpg",
                                image_only_writing,
                            )
                            print(filename)
                            print(
                                "writing_path:" + writing_path + str(count) + ".jpg"
                            )
                            count += 1
                            person_8 += 1
                            file1.close()


print("person 1: " + str(person_1))
print("person 2: " + str(person_2))
print("person 3: " + str(person_3))
print("person 4: " + str(person_4))
print("person 5: " + str(person_5))
print("person 6: " + str(person_6))
print("person 6: " + str(person_7))
print("person 6: " + str(person_8))

import cv2
#import xml.etree.ElementTree as ET
import scipy.io
import os

from chainercv.utils import read_image

person_1 = 0
person_2 = 0
person_3 = 0
person_4 = 0
person_5 = 0

count= 0
#folder_to_consider = ["9--Press_Conference","30--Surgeons","13--Interview","4--Dancing","16--Award_Ceremony","19--Couple","31--Waiter_Waitress","38--Tennis","51--Dresses","54--Rescue"]
annotation = scipy.io.loadmat(r"C:\Users\andre\Desktop\foto and labels\wider_face_split\wider_face_split\wider_face_train.mat")

for i in range(len(annotation['event_list'])):
    event = annotation['event_list'][i,0][0]
    print(event)
    #print(r'C:\Users\andre\Desktop\foto and labels\\WIDER_val_boxes/images/' + event + "\\labels.txt")
    """
    import os
    if (os.path.exists("file.txt") == False):
        f = open("file", "w")
    else:
        print("File Exists") 
    """
     # append mode
    #if event == "0--Parade":
    if 1 == 1: 
    #if event in folder_to_consider:
            
        for j in range(len(annotation['file_list'][i,0])):
            file = annotation['file_list'][i,0][j,0][0]
            filename = "{}.jpg".format(file)
            bboxs = annotation['face_bbx_list'][i,0][j,0]

            
            #if filename == "0_Parade_marchingband_1_45.jpg":
            if 1 == 1: 

                swapped_bbxs = bboxs[:, [1,0,3,2]] #  (y,x,h,w)
                swapped_bbxs[:,2:4] = swapped_bbxs[:,2:4] + swapped_bbxs[:,0:2]
                path_lettura = "C:\\Users\\andre\\Desktop\\foto and labels\\WIDER_train\\WIDER_train\\images\\" + event + "\\" + filename
                path_scrittura = "C:\\Users\\andre\\Desktop\\wider_subset_images\\images\\"
                file_da_scrivere = "C:\\Users\\andre\\Desktop\\wider_subset_images\\images\\" + filename

                
                # Check whether the specified path exists or not
                
                if not os.path.exists(path_scrittura):

                # Create a new directory because it does not exist
                    os.makedirs(path_scrittura)
                #    print("The new directory is created!")

                if (os.path.exists(r'C:\Users\andre\Desktop\wider_subset_images\annots' + "\\" + str(count) + ".txt") == False):
                    file1 = open(r'C:\Users\andre\Desktop\wider_subset_images\annots' + "\\" + str(count) + ".txt", "w")
                    file1.close()
            
                file1 = open(r'C:\Users\andre\Desktop\wider_subset_images\annots' + "\\" + str(count) + ".txt", "a") 
                #file1.write(filename + "\n")

                #print(path_lettura)
                #print(path_scrittura)
                image = cv2.imread(path_lettura)
                image_solo_scrittura = cv2.imread(path_lettura)


                image_area = image_solo_scrittura.shape[0] * image_solo_scrittura.shape[1]
                #test1 = [0,0,0,0]
                #swapped_bbxs_test = swapped_bbxs[0]
                #print(swapped_bbxs)
                #print(filename)
                if not (swapped_bbxs[0][0] == 0 and swapped_bbxs[0][1] == 0   and swapped_bbxs[0][2] == 0 and swapped_bbxs[0][3] == 0):
                        
                    if len(swapped_bbxs) == 1 and person_1 < 125:
                        ok = 1    
                        for k in range(len(swapped_bbxs)):
                            
                            annotation_area = (int(swapped_bbxs[k][3])-int(swapped_bbxs[k][1])) * (int(swapped_bbxs[k][2]) - int(swapped_bbxs[k][0]))
                            #print(swapped_bbxs)
                            #print(annotation_area)
                            #print(image_area)
                            perc = (annotation_area/image_area)*100
                            #print(perc)
                            if perc < 10:
                                if (os.path.exists(r'C:\Users\andre\Desktop\wider_subset_images\annots' + "\\" + str(count) + ".txt") == True):
                                    file1.close()
                                    os.remove(r'C:\Users\andre\Desktop\wider_subset_images\annots' + "\\" + str(count) + ".txt")
                                    ok = 0
                                    break
                            else:
                                cv2.rectangle(image, (swapped_bbxs[k][1], swapped_bbxs[k][0]), (swapped_bbxs[k][3], swapped_bbxs[k][2]), (0, 255, 0),2)
                                
                                file1.write(str(swapped_bbxs[k][1]) + " " + str(swapped_bbxs[k][0]) + " " + str(swapped_bbxs[k][3]) + " " + str(swapped_bbxs[k][2]) + "\n")


                        
                        if ok == 1:
                            cv2.imwrite(path_scrittura+ str(count) +".jpg",image_solo_scrittura)
                            print(filename)
                            print("path_scrittura:" + path_scrittura+ str(count) +".jpg")
                            count += 1
                            person_1 += 1
                            file1.close()

                        

                    elif len(swapped_bbxs) == 2 and person_2 < 125:
                        ok = 1
                        for k in range(len(swapped_bbxs)):
                            
                            annotation_area = (int(swapped_bbxs[k][3])-int(swapped_bbxs[k][1])) * (int(swapped_bbxs[k][2]) - int(swapped_bbxs[k][0]))
                            #print(swapped_bbxs)
                            #print(annotation_area)
                            #print(image_area)
                            perc = (annotation_area/image_area)*100
                            #print(perc)
                            if perc < 5:
                                if (os.path.exists(r'C:\Users\andre\Desktop\wider_subset_images\annots' + "\\" + str(count) + ".txt") == True):
                                    file1.close()
                                    os.remove(r'C:\Users\andre\Desktop\wider_subset_images\annots' + "\\" + str(count) + ".txt")
                                    ok = 0
                                    #print("ci passo minore 3")
                                    break
                            else:
                                cv2.rectangle(image, (swapped_bbxs[k][1], swapped_bbxs[k][0]), (swapped_bbxs[k][3], swapped_bbxs[k][2]), (0, 255, 0),2)
                                
                                file1.write(str(swapped_bbxs[k][1]) + " " + str(swapped_bbxs[k][0]) + " " + str(swapped_bbxs[k][3]) + " " + str(swapped_bbxs[k][2]) + "\n")
                                #print("ci passo maggiore 3")

                        if ok == 1:
                            cv2.imwrite(path_scrittura+ str(count) +".jpg",image_solo_scrittura)
                            print(filename)
                            print("path_scrittura:" + path_scrittura+ str(count) +".jpg")
                            count += 1
                            person_2 += 1
                            file1.close()
                            #print("ci passo scrittura immagine")


                    elif len(swapped_bbxs) == 3 and person_3 < 75:
                        ok = 1
                        for k in range(len(swapped_bbxs)):
                            
                            annotation_area = (int(swapped_bbxs[k][3])-int(swapped_bbxs[k][1])) * (int(swapped_bbxs[k][2]) - int(swapped_bbxs[k][0]))
                            #print(swapped_bbxs)
                            #print(annotation_area)
                            #print(image_area)
                            perc = (annotation_area/image_area)*100
                            #print(perc)
                            if perc < 3:
                                if (os.path.exists(r'C:\Users\andre\Desktop\wider_subset_images\annots' + "\\" + str(count) + ".txt") == True):
                                    file1.close()
                                    os.remove(r'C:\Users\andre\Desktop\wider_subset_images\annots' + "\\" + str(count) + ".txt")
                                    ok = 0
                                    break
                            else:
                                cv2.rectangle(image, (swapped_bbxs[k][1], swapped_bbxs[k][0]), (swapped_bbxs[k][3], swapped_bbxs[k][2]), (0, 255, 0),2)
                                
                                file1.write(str(swapped_bbxs[k][1]) + " " + str(swapped_bbxs[k][0]) + " " + str(swapped_bbxs[k][3]) + " " + str(swapped_bbxs[k][2]) + "\n")




                        if ok == 1:
                            cv2.imwrite(path_scrittura+ str(count) +".jpg",image_solo_scrittura)
                            print(filename)
                            print("path_scrittura:" + path_scrittura+ str(count) +".jpg")
                            count += 1
                            person_3 += 1
                            file1.close()


                    elif len(swapped_bbxs) == 4 and person_4 < 25:
                        ok = 1
                        for k in range(len(swapped_bbxs)):
                            
                            annotation_area = (int(swapped_bbxs[k][3])-int(swapped_bbxs[k][1])) * (int(swapped_bbxs[k][2]) - int(swapped_bbxs[k][0]))
                            #print(swapped_bbxs)
                            #print(annotation_area)
                            #print(image_area)
                            perc = (annotation_area/image_area)*100
                            #print(perc)
                            if perc < 2:
                                if (os.path.exists(r'C:\Users\andre\Desktop\wider_subset_images\annots' + "\\" + str(count) + ".txt") == True):
                                    file1.close()
                                    os.remove(r'C:\Users\andre\Desktop\wider_subset_images\annots' + "\\" + str(count) + ".txt")
                                    ok = 0
                                    break
                            else:
                                cv2.rectangle(image, (swapped_bbxs[k][1], swapped_bbxs[k][0]), (swapped_bbxs[k][3], swapped_bbxs[k][2]), (0, 255, 0),2)
                                
                                file1.write(str(swapped_bbxs[k][1]) + " " + str(swapped_bbxs[k][0]) + " " + str(swapped_bbxs[k][3]) + " " + str(swapped_bbxs[k][2]) + "\n")




                        if ok == 1:
                            cv2.imwrite(path_scrittura+ str(count) +".jpg",image_solo_scrittura)
                            print(filename)
                            print("path_scrittura:" + path_scrittura+ str(count) +".jpg")
                            count += 1
                            person_4 += 1
                            file1.close()

                if (person_1 + person_2 + person_3 + person_4 + person_5) == 350:
                    break

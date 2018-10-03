from  keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

import matplotlib.patches as patches
import keras
### PLOTING ###
class PlotLosses(keras.callbacks.Callback):
    def __init__(self, plot_interval=1):
        self.plot_interval = plot_interval
    
    def on_train_begin(self, logs={}):
        print('Begin training')
        self.i = 0
        self.x = []
        self.total_losses = {}
        self.acc = {}
        self.abserrors = {}
        self.cat_output_loss = {}
        self.bb_loss = {}
        self.ious = {}
        self.logs = []
    
    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        for k,v in logs.items():
            if k in ['loss', 'val_loss']:
                if k not in self.total_losses:
                    self.total_losses[k] = []
                self.total_losses[k].append(v)
            elif 'category_output_loss' in k:
                if k not in self.cat_output_loss:
                    self.cat_output_loss[k] = []
                self.cat_output_loss[k].append(v)
            elif 'bounding_box_loss' in k:
                if k not in self.bb_loss:
                    self.bb_loss[k] = []
                self.bb_loss[k].append(v)
            elif 'acc' in k:
                if k not in self.acc:
                    self.acc[k] = []
                self.acc[k].append(v)
            elif 'error' in k:
                if k not in self.abserrors:
                    self.abserrors[k] = []
                self.abserrors[k].append(v)
            elif 'iou' in k:
                if k not in self.ious:
                    self.ious[k] = []
                self.ious[k].append(v)
                
        self.logs.append(logs)
        self.x.append(self.i)
        self.i += 1
        if (epoch%self.plot_interval==0):
            clear_output(wait=True)
            to_plot = [self.total_losses, self.cat_output_loss, self.bb_loss, self.abserrors, self.acc, self.ious]
            not_empty = []
            subplots = 0
            for pl in to_plot:
                subplots = subplots + 1*(len(pl)>0)
                if len(pl) > 0:
                    not_empty.append(pl)
            rows = int(np.ceil(subplots / 2))
            f, axs = plt.subplots(rows, 2, sharex=True, figsize=(20,5*rows))
            axs = axs.flatten()
            for i, ne in enumerate(not_empty):
                for k,v in ne.items():
                    if 'val' in k:
                        axs[i].plot(self.x, v, label=k, ls='-.', color='b')
                    else:
                        axs[i].plot(self.x, v, label=k, color='r')        
                    
                axs[i].legend()
            plt.show()
            
def getBB_area(bb):
    IntersectionArea = (bb[:,2] - bb[:,0])*(bb[:,3] - bb[:,1])
    return IntersectionArea

def getIUO(bb1, bb2, from_center_to_box = False):
    if from_center_to_box:
        bb1 = np.array([
             bb1[:,0] - bb1[:,2]/2, 
             bb1[:,1] - bb1[:,3]/2,
             bb1[:,0] + bb1[:,2]/2, 
             bb1[:,1] + bb1[:,3]/2,]).T
        bb2 = np.array([
             bb2[:,0] - bb2[:,2]/2, 
             bb2[:,1] - bb2[:,3]/2,
             bb2[:,0] + bb2[:,2]/2, 
             bb2[:,1] + bb2[:,3]/2,]).T

    intersection_bb = np.array([np.vstack([bb1[:,0], bb2[:,0]]).max(axis=0),
        np.vstack([bb1[:,1], bb2[:,1]]).max(axis=0),
        np.vstack([bb1[:,2], bb2[:,2]]).min(axis=0),
        np.vstack([bb1[:,3], bb2[:,3]]).min(axis=0)]).T
    no_intersec = 1*(intersection_bb[:,3]-intersection_bb[:,1]>0)*(intersection_bb[:,2]-intersection_bb[:,0]>0)
    intersection_bb = (intersection_bb.T * no_intersec).T
    IntersectionArea = no_intersec*getBB_area(intersection_bb)
    IOU = IntersectionArea/(getBB_area(bb1) + getBB_area(bb2) - IntersectionArea)
    return IOU, intersection_bb
            
def plot_batch(generator, model=None, count = 10):
    pred_bounding_boxes = None
    n_classes = generator.generator.num_classes
    if 'world' in generator.generator.class_indices:
        n_classes = n_classes - 1
    batch = next(generator)
    images_batch = batch[0]
    if generator.concat_output:
        annotations = batch[1]
    else:
        annotations = np.hstack([batch[1][0], batch[1][1]])

    if model is not None:
        predictions_ = model.predict_on_batch(images_batch)
        if generator.concat_output:
            predictions = predictions_
        else:
            predictions = np.hstack([predictions_[0], predictions_[1]])
        
        bounding_boxes_norm = predictions[:,n_classes:n_classes+4]
        pred_bounding_boxes = np.array([bounding_boxes_norm[:,0] - bounding_boxes_norm[:,2]/2, 
                                   bounding_boxes_norm[:,1] - bounding_boxes_norm[:,3]/2,
                                   bounding_boxes_norm[:,0] + bounding_boxes_norm[:,2]/2, 
                                   bounding_boxes_norm[:,1] + bounding_boxes_norm[:,3]/2,]).T
    
    bounding_boxes_norm = annotations[:,n_classes:n_classes+4]
    bounding_boxes = np.array([bounding_boxes_norm[:,0] - bounding_boxes_norm[:,2]/2, 
                               bounding_boxes_norm[:,1] - bounding_boxes_norm[:,3]/2,
                               bounding_boxes_norm[:,0] + bounding_boxes_norm[:,2]/2, 
                               bounding_boxes_norm[:,1] + bounding_boxes_norm[:,3]/2,]).T    
    for image_index in range(count):
        image_ground_truth = annotations[image_index]
        
        if model is not None:
            image_predictions = predictions[image_index]
            if generator.has_world:
                if image_ground_truth[-1] == int(np.round(image_predictions[-1])):
                    print('Confidence OK:', image_predictions[-1])
                else:
                    print('Confidence Failed:', image_predictions[-1])
            if (generator.has_world and image_ground_truth[-1] == 1) or not generator.has_world:    
                if np.argmax(image_ground_truth[:n_classes]) == np.argmax(image_predictions[:n_classes]):
                    print('Class OK:', np.argmax(image_ground_truth[:n_classes]))
                else:
                    print('Class Failed:', np.argmax(image_ground_truth[:n_classes]), np.argmax(image_predictions[:n_classes]))
                iou, _ = getIUO(image_ground_truth[n_classes:n_classes+4].reshape(1, -1), 
                             image_predictions[n_classes:n_classes+4].reshape(1, -1), from_center_to_box=True)
                print('IOU:', iou)
        else:
            print((annotations[image_index]*100).astype(int)/100)
        #if class_vector.sum() == 1:
        #    print(generator.idx_2_class_id[np.argmax(class_vector)])
        
        f, ax = plt.subplots(1,1)
        ax.imshow(images_batch[image_index])
        im_w = images_batch[image_index].shape[1]
        im_h = images_batch[image_index].shape[0]
        
        bounding_box = bounding_boxes[image_index]
        bounding_box[0] = bounding_box[0]*im_w
        bounding_box[2] = bounding_box[2]*im_w
        bounding_box[1] = bounding_box[1]*im_h
        bounding_box[3] = bounding_box[3]*im_h
        rect_gt = patches.Rectangle(bounding_box[:2],
                                        (bounding_box[2]-bounding_box[0]),
                                        (bounding_box[3]-bounding_box[1]),
                                        linewidth=2, edgecolor='r',facecolor='none')
        ax.add_patch(rect_gt)
        if pred_bounding_boxes is not None:
            bounding_box = pred_bounding_boxes[image_index]
            bounding_box[0] = bounding_box[0]*im_w
            bounding_box[2] = bounding_box[2]*im_w
            bounding_box[1] = bounding_box[1]*im_h
            bounding_box[3] = bounding_box[3]*im_h
            rect_gt = patches.Rectangle(bounding_box[:2],
                                            (bounding_box[2]-bounding_box[0]),
                                            (bounding_box[3]-bounding_box[1]),
                                            linewidth=2, edgecolor='y',facecolor='none')
            ax.add_patch(rect_gt)
        
        plt.show()
        

class GeneratorMultipleOutputs(Sequence):
    def __init__(self, annotations_dict, folder, batch_size, flip = 'no_flip', concat_output=True, get_filenames = False, target_size=(375, 500), classes=None, no_ext_idx=True, width_heigh_exp = False):
        # flip = {no_flip, always, random}
        # concat_output: classes + bounding box + confidence, todo en unico np.array
        # no_ext_idx: los keys del anotation son el filename pero sin extención (Se lo tengo que quitar entonces)
        # width_heigh_exp: La bounding box viene con witdh y height en las posiciones 3 y 4. Eso es asi en COCO
        self.width_heigh_exp = width_heigh_exp
        self.concat_output = concat_output
        self.flip = flip
        self.no_ext_idx = no_ext_idx
        self.get_filenames = get_filenames
        np.random.seed(seed=40)
        self.annotations_dict = annotations_dict
        datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=False)
        self.generator = datagen.flow_from_directory(
            classes = classes,
            directory=folder,
            target_size=target_size,
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        self.idx_2_class_id = {v:k for k,v in self.generator.class_indices.items()}
        self.has_world = 'world' in self.generator.class_indices
    def get_image_object_center(self):
        bboxes = []
        batch_index = self.generator.batch_index
        if self.generator.batch_index == 0:
            batch_index = self.__len__()
        batch_filenames = np.array(self.generator.filenames)[self.generator.index_array][(batch_index-1)*self.generator.batch_size:batch_index*self.generator.batch_size]
        classes = np.array(self.generator.classes)[self.generator.index_array][(batch_index-1)*self.generator.batch_size:batch_index*self.generator.batch_size]
        annot_dicts = []
        box_widths = []
        box_heights = []
        centerXs = []
        centerYs = []
        object_detected_arr = []
        for i, filename in enumerate(batch_filenames):
            class_id = self.idx_2_class_id[classes[i]] #filename.split('/')[0]
            image_idx = filename.split('/')[-1]
            if self.no_ext_idx:
                image_idx = image_idx.split('.')[0]
            if class_id == 'world':
                annot_dict = {'width':-1, 'height':-1, 'bounding_boxes':[[0,0,0,0]]}
                object_detected_arr.append(0)
            else:
                annot_dict = self.annotations_dict[class_id][image_idx]
                object_detected_arr.append(1)
                
            img_width = annot_dict['width']
            img_height = annot_dict['height']
            bounding_box = annot_dict['bounding_boxes'][0]
            if self.width_heigh_exp:
                box_width = bounding_box[2]
                box_height = bounding_box[3]
            else:
                box_width = bounding_box[2] - bounding_box[0]
                box_height = bounding_box[3] - bounding_box[1]
            centerX = (bounding_box[0]+(box_width)/2)/img_width
            centerY = (bounding_box[1]+(box_height)/2)/img_height
            box_widths.append(box_width/img_width)
            box_heights.append(box_height/img_height)
            centerXs.append(centerX)
            centerYs.append(centerY)
            annot_dicts.append(annot_dict)
        return np.array(centerXs), np.array(centerYs), np.array(box_widths), np.array(box_heights), batch_filenames, annot_dicts, np.array([object_detected_arr])
    def __len__(self):
        return int(np.ceil(self.generator.samples / float(self.generator.batch_size)))
    def __getitem__(self, idx):
        data = next(self.generator)
        centerX, centerY, width, height, batch_filenames, annot_dicts, object_detected_arr = self.get_image_object_center()
        if self.flip == 'random':
            inices_to_flip = np.random.randint(0, 2, data[0].shape[0]).nonzero()
            data[0][inices_to_flip] = np.flip(data[0][inices_to_flip], axis = 2)
            centerX[inices_to_flip] = 1 - centerX[inices_to_flip]
        elif self.flip == 'always':
            data[0][:] = np.flip(data[0][:], axis = 2)
            centerX = 1 - centerX
        if self.has_world:
            # Borro la clase world de la salida
            classes_array = np.delete(data[1], [self.generator.class_indices['world']], axis=1)
        else:
            classes_array = data[1]
        
        centerX = centerX*(object_detected_arr).reshape(-1)
        centerY = centerY*(object_detected_arr).reshape(-1)
        width = width*(object_detected_arr).reshape(-1)
        height = height*(object_detected_arr).reshape(-1)
        
        if self.concat_output:
            if self.has_world:
                output = np.hstack([classes_array, np.array([centerX, centerY, width, height]).T, object_detected_arr.T])
            else:
                output = np.hstack([classes_array, np.array([centerX, centerY, width, height]).T])
                
        else:
            if self.has_world:
                output = [classes_array, np.array([centerX, centerY, width, height]).T, object_detected_arr.T]
            else:
                output = [classes_array, np.array([centerX, centerY, width, height]).T]
        
            
        if self.get_filenames:
            return (data[0], output, batch_filenames, annot_dicts, object_detected_arr)
        else:    
            return (data[0], output)
    def __next__(self):
        return self.__getitem__(0)
    def __iter__(self):
        return self
    
    
### MODELS ###

from keras.layers import Activation, Dropout, Dense, Input, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Concatenate, GlobalMaxPooling2D, Flatten
from keras.models import Model
from keras.constraints import max_norm

def get_conv_layer(x, filters = 32, filter_size = (3,3), pool_size=(2,2)):
    conv = Conv2D(filters, filter_size)(x)
    BN = BatchNormalization()(conv)
    act = Activation('relu')(BN)
    out = MaxPooling2D(pool_size=pool_size)(act)
    # DO1 = Dropout(0.25)(maxPool1)
    return out

def get_simple_model_common_part(input_shape=(375, 500, 3)):
    x = Input(shape=(375, 500, 3))
    l1 = get_conv_layer(x, filters = 32, filter_size = (3,3), pool_size=(2,2))
    
    l2 = get_conv_layer(l1, filters = 64, filter_size = (3,3), pool_size=(2,2))
    
    l3 = get_conv_layer(l2, filters = 128, filter_size = (3,3), pool_size=(2,2))
    
    l4 = get_conv_layer(l3, filters = 256, filter_size = (3,3), pool_size=(2,2))
    
    l5 = get_conv_layer(l4, filters = 512, filter_size = (3,3), pool_size=(2,2))

    GAP = GlobalAveragePooling2D()(l5)
    model = Model(x, GAP)
    return model

def get_simple_model(input_shape=(375, 500, 3), n_classes=5, dropout_rate_1 = 0.5, dropout_rate_2 = 0.1):
    base_model = get_simple_model_common_part(input_shape=input_shape)
    classification = Dense(n_classes, activation='softmax', name='category_output', kernel_constraint=max_norm(1.))(Dropout(dropout_rate_1)(base_model.output))
    bounding_box = Dense(4, name='bounding_box', kernel_constraint=max_norm(2.))(Dropout(dropout_rate_2)(base_model.output))
    model = Model(inputs=base_model.input, outputs=[classification, bounding_box])
    return model

from keras.applications.vgg16 import VGG16
from keras.layers import Reshape

def get_VGG16_no_dense(n_classes = 5, dropout_rate_1 = 0.5, dropout_rate_2 = 0.25, N_trainable = 19, input_shape=(375, 500,3)):
    modelVGG16 = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    output = modelVGG16.output
    last_h = output.shape[1].value
    last_w = output.shape[2].value
    classification = Flatten(name='category_output')(Conv2D(filters = n_classes, kernel_size = (last_h, last_w), activation='softmax')(output))
    bounding_box = Flatten(name='bounding_box')(Conv2D(filters = 4, kernel_size = (last_h, last_w), activation=None)(output))
    
    model = Model(inputs=modelVGG16.input, outputs=[classification, bounding_box])
    for layer in model.layers[N_trainable:]:
        layer.trainable = True
    for layer in model.layers[:N_trainable]:
        layer.trainable = False
    return model

def get_VGG16(n_classes = 5, dropout_rate_classif = 0.5, dropout_bbox = 0.5, N_trainable = 19, BN = False):
    modelVGG16 = VGG16(include_top=False, weights='imagenet')
    if BN:
        GAP = GlobalAveragePooling2D()(BatchNormalization()(modelVGG16.output))
    else:
        GAP = GlobalAveragePooling2D()(modelVGG16.output)
        
    if dropout_rate_classif>0:
        classification = Dense(n_classes, 
                               activation='softmax', 
                               name='category_output', 
                               kernel_constraint=max_norm(1.))(Dropout(dropout_rate_classif)(GAP))
    else:
        classification = Dense(n_classes, 
                               activation='softmax', 
                               name='category_output', 
                               kernel_constraint=max_norm(1.))(GAP)
        
    if dropout_bbox>0:
        bounding_box = Dense(4, 
                             name='bounding_box', 
                             kernel_constraint=max_norm(2.))(Dropout(dropout_bbox)(GAP))
    else:
        bounding_box = Dense(4, 
                             name='bounding_box', 
                             kernel_constraint=max_norm(2.))(GAP)
        
    model = Model(inputs=modelVGG16.input, outputs=[classification, bounding_box])
    for layer in model.layers[N_trainable:]:
        layer.trainable = True
    for layer in model.layers[:N_trainable]:
        layer.trainable = False
    return model

def get_VGG16_world(n_classes = 5, input_shape=(375, 500, 3), dropout_class = 0.5, dropout_confidence = 0.5, dropout_bbox = 0.5, N_trainable = 19, activation_class='softmax', activation_bbox=None, activation_confidence='sigmoid'):
    modelVGG16 = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    GAP = GlobalAveragePooling2D()(BatchNormalization()(modelVGG16.output))
    
    classification = Dense(n_classes, activation=activation_class, name='category_output', kernel_constraint=max_norm(1.))(Dropout(dropout_class)(GAP))
    bounding_box = Dense(4, activation=activation_bbox, name='bounding_box', kernel_constraint=max_norm(2.))(Dropout(dropout_bbox)(GAP))
    confidence = Dense(1, activation=activation_confidence, name='obj_confidence', kernel_constraint=max_norm(2.))(Dropout(dropout_confidence)(GAP))
    all_outs = Concatenate(name='concatenated_outputs')([classification, bounding_box, confidence])
    model = Model(inputs=modelVGG16.input, outputs=[all_outs])
    for layer in model.layers[N_trainable:]:
        layer.trainable = True
    for layer in model.layers[:N_trainable]:
        layer.trainable = False
    return model
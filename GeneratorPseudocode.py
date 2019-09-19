class GeneratorMultipleOutputs(Sequence):
    def __init__(self, annotations_dict, folder, batch_size, flip = 'no_flip', concat_output=True, target_size=(375, 500), classes=None):
        self.concat_output = concat_output, self.flip = flip, self.annotations_dict = annotations_dict
        np.random.seed(seed=40)
        datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=False)
        self.generator = datagen.flow_from_directory(
            classes = classes, directory=folder, target_size=target_size, color_mode="rgb", batch_size=batch_size,
            class_mode="categorical", shuffle=True, seed=42
        )
        
    def get_image_object_center(self):
        # Obtengo bounding boxes de un batch,  Calculo centros y anchos luego de augmentation,  Y las devuelvo
        return centerXs, centerYs, box_widths, box_heights
    
    def __len__(self):
        return int(np.ceil(self.generator.samples / float(self.generator.batch_size)))
    
    def __getitem__(self, idx):
        data = next(self.generator)
        centerX, centerY, width, height = self.get_image_object_center()
        # object_detected_arr es un array que tiene un 0 si la imagen pertenece a la clase word 
        # (El calculo no esta aca para hacer el código más sencillo)
        
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
        return (data[0], output)
    
    def __next__(self):
        return self.__getitem__(0)
    
    def __iter__(self):
        return self
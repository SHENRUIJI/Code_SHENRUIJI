四部分，yolo代码，其他代码两个，数据预处理代码
展示模型运行成功截图
展示引言

## Ниже приведён рисунок, показывающий, что моя модель работает корректно.

Ниже показан YOLOv8n:

![YOLOv8n](./run_results/yolov8.png)


Ниже показан YOLOv8n+SE:

![YOLOv8n+SE](./run_results/yolov8se.png)

Ниже показан YOLOv11n:

![YOLOv11n](./run_results/yolov11.png)

Ниже показан Resnet34:

![Resnet34](./run_results/resnet.png)

Ниже показан SE-Resnet34:

![SE-Resnet34](./run_results/seresnet.png)

Ниже показан Densenet:

![Densenet](./run_results/densenet.png)

Ниже показан Deit-tiny:

![Deit-tiny](./run_results/deit.png)

Ниже показан EfficientFormer-L3:

![EfficientFormer-L3](./run_results/efficientformer.png)

## Предварительная обработка данных：

**cleanup.py**  
Скрипт для очистки данных и исправления несоответствий между изображениями и метками.

**yolo2coco.py**  
Этот скрипт преобразует разметку YOLO в формат COCO.

**yolo2voc.py**  
Этот скрипт преобразует разметку YOLO в формат VOC.

**check_yolo_labels.py**  
Проверка целостности разметки в наборе данных.

**rename_yolo.py**  
Пакетное переименование изображений и файлов разметки YOLO.

**rename.py**  
Пакетное переименование файлов изображений

**augment_yolo.py**  
Аугментация набора данных YOLO

**crop_imgs.py**  
Обрезка изображений по разметке YOLO

**update_labels.py**  
Изменение меток классов YOLO.

**splite.py**  
Итеративное разделение обучающего и проверочного наборов данных

**splite.py**  
Подсчёт количества объектов по классам.

**compress_imgs.py**  
Сжатие качества изображений.

## YOLO

**yolo/train8.py**  
Код для обучения YOLOv8.

**yolo/train11.py**  
Код для обучения YOLOv11.

**yolo/train8se.py**  
Код для обучения YOLOv8+SE.

**yolo/ultralytics/cfg/models/v8/yolov8.ymal**  
Файл конфигурации модели YOLOv8.

**yolo/ultralytics/cfg/models/v8/SEAtt_yolov8.yaml**  
Файл конфигурации модели YOLOv8+SE.

**yolo/ultralytics/cfg/models/11/yolo11.ymal**  
Файл конфигурации модели YOLOv11.

**yolo/ultralytics/cfg/datasets/crop.yaml**  
Файл конфигурации набора данных YOLO.

## Основные файлы других моделей

**GetAnnot.py**  
Список данных для набора генерации и тестирования.

**Resnet.py**  
Файл конфигурации обучения модели.

**seresnet.py**  
Файл конфигурации обучения модели

**Densenet.py**  
Файл конфигурации обучения модели

**datas/annotations.txt**  
Файл сопоставления классов и меток

**datas/train.txt**  
Список данных обучающего набора

**datas/test.txt**  
Список данных тестового набора


## Ссылка на репозиторий GitHub：
@misc{2023mmpretrain,
    title={OpenMMLab's Pre-training Toolbox and Benchmark},
    author={MMPreTrain Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpretrain}},
    year={2025}
}

@misc{ultralytics,
  author       = {Ultralytics},
  title        = {Ultralytics},
  year         = 2024,
  howpublished = {\url{https://github.com/ultralytics/ultralytics}},
  note         = {Accessed: 2024}
}

@misc{Awesome-Backbones,
  author       = {Fafa-DL},
  title        = {Awesome-Backbones},
  year         = 2024,
  howpublished = {\url{https://github.com/Fafa-DL/Awesome-Backbones}},
  note         = {Accessed: 2025}
}

from ultralytics import YOLO

def treina_def_mod_yolo_data_augmentation():
    # Carrega o modelo 'yolo11n.pt'
    model = YOLO('yolo11n.pt')
    print("A iniciar treino...")

    # Informações 
    model.info()

    # Treino
    results = model.train(cfg="conf.yaml")
    print("Treino concluído.")

    # Validação
    val_results = model.val(project='runs/detect', name='val')

if __name__ == "__main__":
    treina_def_mod_yolo_data_augmentation()



























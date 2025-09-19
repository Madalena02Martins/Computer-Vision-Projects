import tkinter as tk 
from tkinter import ttk
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import random
import os
import shutil
from ultralytics import YOLO
import subprocess
import sys
import yaml
import threading

# Janela principal
win = Tk()
win.title("Trabalho final")
ws, hs = win.winfo_screenwidth(), win.winfo_screenheight()
win.geometry(f'{ws}x{hs}+0+0')

# Variáveis globais
img_pil = None
filename = None
error_label = None
after_id = None
cap = None

# Declaração de caminhos
data_path = os.path.join(os.getcwd(), 'data.yaml')
base_path = os.path.join(os.getcwd(), 'dados')
results_path = os.path.join(os.getcwd(), 'runs', 'detect')

# Tamanho máximo permitido para os canvas
max_canvas_width = 700
max_canvas_height = 500

mensagem1_label = tk.Label(win, text="", font=("Arial", 11))
mensagem1_label.place(x=370, y=630, anchor=NW)

# Mostra uma mensagem temporária na interface
def mostrar_mensagem(texto, cor="blue", duracao=3000):
    mensagem1_label.config(text=texto, fg=cor)
    win.update()
    win.after(duracao, lambda: mensagem1_label.config(text=""))

# Mostra uma mensagem permanente na interface
def mostrar_mensagem_constante(texto, cor="blue"):
    mensagem1_label.config(text=texto, fg=cor)
    win.update()

# Executa o script de treino de modelo usando subprocesso
def correr_treino():
    mostrar_mensagem_constante("A treinar... Por favor, aguarde.")
    subprocess.run([sys.executable, "train.py"])
    mostrar_mensagem("Subprocesso concluído.", cor="green")
    print("Subprocesso concluído.")

# Cria o ficheiro data.yaml com caminhos e nomes das classes
def criar_data_yaml(caminho_classes='classes.txt', caminho_saida='data.yaml'):
    try:
        with open(caminho_classes, 'r', encoding='utf-8') as f:
            classes = [linha.strip() for linha in f if linha.strip()]
    except FileNotFoundError:
        print(f"Erro: o ficheiro '{caminho_classes}' não foi encontrado.")
        return

    conteudo_yaml = {
        'train': './dados/train',
        'val': './dados/val',
        'names': {i: nome for i, nome in enumerate(classes)}
    }

    with open(caminho_saida, 'w', encoding='utf-8') as f:
        yaml.dump(conteudo_yaml, f, allow_unicode=True, sort_keys=False)

    mostrar_mensagem("Criação do ficheiro data.yaml concluída", cor="green")
    win.after(2000) 
    print(f"Ficheiro '{caminho_saida}' criado com sucesso.")

# Atualiza ou cria o ficheiro conf.yaml com os parâmetros de treino
def atualizar_config():
    mostrar_mensagem("Atualização do ficheiro conf.yaml")
    win.after(2000)  
    if not os.path.exists("conf.yaml"):
        config_default = {
            "epochs": 50,
            "batch": 16,
            "hsv_s": 0.5,
            "hsv_v": 0.5,
            "lr0": 0.001,
            "lrf": 0.01,
            "data": "",
            "mode": "train",
            "model": "yolo11n.pt"
        }
        with open("conf.yaml", "w", encoding='utf-8') as f:
            yaml.safe_dump(config_default, f, allow_unicode=True)

    with open("conf.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        print(config)

    try:
        config["epochs"] = int(e_entry.get("1.0", "end").strip())
        config["batch"] = int(b_entry.get("1.0", "end").strip())
        config["hsv_s"] = float(hsv_s_entry.get("1.0", "end").strip())
        config["hsv_v"] = float(hsv_v_entry.get("1.0", "end").strip())
        config["lr0"] = float(lr0_entry.get("1.0", "end").strip())
        config["lrf"] = float(lrf_entry.get("1.0", "end").strip())
    except ValueError:
        show_error_message("Erro: Verifica se todos os campos foram preenchidos corretamente.")
        return

    config["data"] = data_path.replace("\\", "\\\\")
    print(config["data"])

    with open("conf.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)

    print("Configuração atualizada com sucesso!")

# Verifica a existência do ficheiro data.yaml
def verificar_ficheiro_yaml():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(base_dir, 'data.yaml')

    if not os.path.isfile(yaml_path):
        show_error_message("Erro: ficheiro 'data.yaml' não encontrado. Verifique a Criação do ficheiro data.yaml.")
        return None
    
    return yaml_path

# Verifica a existência dos diretórios 'train' e 'val'
def obter_diretorios_dados():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dados_dir = os.path.join(base_dir, 'dados')

    train_dir = os.path.join(dados_dir, 'train')
    val_dir = os.path.join(dados_dir, 'val')

    if not os.path.isdir(train_dir):
        show_error_message(f"Erro: pasta 'train' não encontrada. Realize a preparação do dataset.")
        return None, None
    if not os.path.isdir(val_dir):
        show_error_message(f"Erro: pasta 'val' não encontrada. Realize a preparação do dataset.")
        return None, None

    return train_dir, val_dir

# Interrompe a captura e exibição de vídeo
def parar_video():
    global after_id, cap
    if after_id is not None:
        win.after_cancel(after_id)
        after_id = None
    if cap is not None:
        cap.release()
        cap = None
    canvas1.delete("all")
    canvas2.delete("all")

# Mostra mensagem de erro na interface por tempo limitado
def show_error_message(message):
    global error_label
    if error_label is not None:
        error_label.place_forget()
    error_label = Label(win, text=message, font=("Arial", 10), foreground="red")
    error_label.place(x=370, y=640, anchor=W)
    win.after(10000, hide_error_label)

# Esconde a label de erro da interface
def hide_error_label():
    global error_label
    if error_label is not None:
        error_label.place_forget()
        error_label = None

# Restaura cor de fundo do widget ao valor padrão (usado em eventos)
def reset_bg(event):
    event.widget.config(bg='white')

# Calcula o novo tamanho de uma imagem para caber dentro do canvas
def nova_img(largura_img, altura_img, max_largura, max_altura):
    scale_w = max_largura / largura_img
    scale_h = max_altura / altura_img
    scale = min(scale_w, scale_h)
    largura_nova = round(largura_img * scale)
    altura_nova = round(altura_img * scale)
    return (largura_nova, altura_nova)

# Permite ao utilizador escolher uma imagem e exibi-la no canvas1
def esc_img():
    global img_pil, imgc, canvas1, filename
    filename = filedialog.askopenfilename(title="Escolha de ficheiro")
    if filename:
        img_pil = Image.open(filename).convert("RGB")
        largura_img, altura_img = img_pil.size
        largura_nova, altura_nova = nova_img(largura_img, altura_img, max_canvas_width, max_canvas_height)
        img_pil_resized = img_pil.resize((largura_nova, altura_nova), Image.Resampling.LANCZOS)
        imgc = ImageTk.PhotoImage(img_pil_resized)
        canvas1.config(width=largura_nova, height=altura_nova)
        canvas1.delete("all")
        canvas1.create_image(0, 0, anchor=tk.NW, image=imgc)
        win.update_idletasks()

# Copia imagens e seus rótulos para os diretórios corretos de treino, validação ou teste
def copia_ficheiros(base_path, images_path, labels_path, files, set_type):
    for file in files:
        shutil.copy(os.path.join(images_path, file), os.path.join(base_path, set_type, 'images'))
        label_file = file.rsplit('.', 1)[0] + '.txt'
        shutil.copy(os.path.join(labels_path, label_file), os.path.join(base_path, set_type, 'labels'))

# Prepara o dataset dividindo os dados em treino, validação e teste
def preparadataset(base_path):
    mostrar_mensagem("Preparação do dataset iniciada")
    win.after(2000)  
    random.seed(42)
    images_path = os.path.join(base_path, 'images')
    labels_path = os.path.join(base_path, 'labels')
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15
    for set_type in ['train', 'val', 'test']:
        for content_type in ['images', 'labels']:
            os.makedirs(os.path.join(base_path, set_type, content_type), exist_ok=True)
    all_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    random.shuffle(all_files)
    total_files = len(all_files)
    train_end = int(train_ratio * total_files)
    val_end = train_end + int(val_ratio * total_files)
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]
    copia_ficheiros(base_path, images_path, labels_path, train_files, 'train')
    copia_ficheiros(base_path, images_path, labels_path, val_files, 'val')
    copia_ficheiros(base_path, images_path, labels_path, test_files, 'test')
    mostrar_mensagem("Preparação do dataset terminada", cor="green")
    win.after(2000)  

# Mostra os resultados do treino (matriz de confusão e gráfico) e pergunta se deseja guardar o modelo
def mostrar_resultados():
    global results_path
    pastas_train = [d for d in os.listdir(results_path)
                    if 'train' in d and os.path.isdir(os.path.join(results_path, d))]
    pastas_train.sort()

    if pastas_train:
        ultimo_treino = pastas_train[-1]
        run_dir = os.path.join(results_path, ultimo_treino)
    else:
        show_error_message("Nenhuma pasta com 'train' no nome foi encontrada.")
        return

    best_model_path = os.path.join(run_dir, 'weights', 'best.pt')

    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, './best.pt')
        mostrar_mensagem("Modelo 'best.pt' copiado para a pasta atual.")
    else:
        show_error_message("Ficheiro 'best.pt' não encontrado no diretório de treino.")
        return

    img1_path = os.path.join(run_dir, 'confusion_matrix.png')
    img2_path = os.path.join(run_dir, 'results.png')

    nova_janela = Toplevel(win)
    nova_janela.title("Resultado do Treino")
    nova_janela.geometry("1500x750")

    img1 = Image.open(img1_path).resize((600, 450), Image.Resampling.LANCZOS)
    img2 = Image.open(img2_path).resize((820, 450), Image.Resampling.LANCZOS)
    img1_tk = ImageTk.PhotoImage(img1)
    img2_tk = ImageTk.PhotoImage(img2)
    nova_janela.img1_tk = img1_tk
    nova_janela.img2_tk = img2_tk
    tk.Label(nova_janela, image=img1_tk).place(x=20, y=20, anchor=NW)
    tk.Label(nova_janela, image=img2_tk).place(x=650, y=20, anchor=NW)

    tk.Label(nova_janela, text="Deseja guardar o modelo treinado?", font=("Arial", 12, "bold")).place(x=600, y=500, anchor=NW)

    mensagem_label = tk.Label(nova_janela, text="", font=("Arial", 11))
    mensagem_label.place(x=740, y=610, anchor=CENTER)

    def guardar_modelo():
        if os.path.exists('./best.pt'):
            mensagem_label.config(text="Modelo guardado com sucesso!", fg="green")
        else:
            mensagem_label.config(text="Erro: 'best.pt' não encontrado.", fg="red")
        print('Modelo guardado com sucesso!')

    def nao_guardar_modelo():
        if os.path.exists('./best.pt'):
            os.remove('./best.pt')
        mensagem_label.config(text="Modelo não foi guardado.", fg="blue")
        print('Modelo não foi guardado.')

    Button(nova_janela, text="Sim", command=guardar_modelo, width=10).place(x=550, y=550, anchor=NW)
    Button(nova_janela, text="Não", command=nao_guardar_modelo, width=10).place(x=850, y=550, anchor=NW)
    Button(nova_janela, text="Fechar", command=nova_janela.destroy).place(x=700, y=650, anchor=NW)

# Testa a imagem selecionada com o modelo treinado e exibe o resultado no canvas2
def testar_imagem_com_modelo():    
    global filename, canvas2, imgc2
    model = YOLO('best.pt')
    confidence_threshold = 0.5 
    results = model(filename, conf=confidence_threshold)
    results[0].save(filename="resultado.jpg")
    result_img = Image.open("resultado.jpg").convert("RGB")
    largura_img, altura_img = result_img.size
    largura_nova, altura_nova = nova_img(largura_img, altura_img, max_canvas_width, max_canvas_height)
    result_img_resized = result_img.resize((largura_nova, altura_nova), Image.Resampling.LANCZOS)
    imgc2 = ImageTk.PhotoImage(result_img_resized)
    canvas2.config(width=largura_nova, height=altura_nova)
    canvas2.delete("all")
    canvas2.create_image(0, 0, anchor=tk.NW, image=imgc2)
    win.update_idletasks()

# Captura vídeo da webcam, aplica o modelo em tempo real e exibe os frames originais e anotados
def detetar_video_camera():
    global after_id, cap
    cap = cv2.VideoCapture(0)
    model = YOLO('best.pt')

    def atualizar_frame():
        global after_id
        ret, frame = cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil_original = Image.fromarray(frame_rgb)
        img_resized1 = img_pil_original.resize((max_canvas_width, max_canvas_height), Image.Resampling.LANCZOS)
        img_tk1 = ImageTk.PhotoImage(img_resized1)
        canvas1.img_tk = img_tk1
        canvas1.create_image(0, 0, anchor=tk.NW, image=img_tk1)
        results = model(frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img_pil_annotated = Image.fromarray(annotated_rgb)
        img_resized2 = img_pil_annotated.resize((max_canvas_width, max_canvas_height), Image.Resampling.LANCZOS)
        img_tk2 = ImageTk.PhotoImage(img_resized2)
        canvas2.img_tk = img_tk2
        canvas2.create_image(0, 0, anchor=tk.NW, image=img_tk2)
        after_id = win.after(30, atualizar_frame)

    atualizar_frame()

# Oculta todos os campos de entrada e botão, usado para "limpar" a interface
def hide_fields():
    b_label.place_forget()
    b_entry.place_forget()
    e_label.place_forget()
    e_entry.place_forget()
    hsv_s_label.place_forget()
    hsv_s_entry.place_forget()
    hsv_v_label.place_forget()
    hsv_v_entry.place_forget()
    lr0_label.place_forget()
    lr0_entry.place_forget()
    lrf_label.place_forget()
    lrf_entry.place_forget()
    btn1.place_forget()

# Executada ao selecionar uma opção na listbox; ajusta a interface conforme a escolha
def select_listbox():
    hide_error_label()  # Esconde mensagens de erro anteriores
    op = listbox.curselection()
    if not op:
        return
    op = op[0]

    hide_fields()  # Oculta todos os campos antes de exibir os necessários

    if op == 0:
        # Limpa os campos e redefine os fundos para branco
        b_entry.delete("1.0", tk.END)
        e_entry.delete("1.0", tk.END)
        hsv_s_entry.delete("1.0", tk.END)
        hsv_v_entry.delete("1.0", tk.END)
        lr0_entry.delete("1.0", tk.END)
        lrf_entry.delete("1.0", tk.END)
        b_entry.configure(bg='white')
        e_entry.configure(bg='white')
        hsv_s_entry.configure(bg='white')
        hsv_v_entry.configure(bg='white')
        lr0_entry.configure(bg='white')
        lrf_entry.configure(bg='white')
        canvas1.delete("all")
        canvas2.delete("all")
        canvas1.config(width=max_canvas_width, height=max_canvas_height)
        canvas2.config(width=max_canvas_width, height=max_canvas_height)
        parar_video()

    if op == 1:
        # Mesmo comportamento do caso 0
        b_entry.delete("1.0", tk.END)
        e_entry.delete("1.0", tk.END)
        hsv_s_entry.delete("1.0", tk.END)
        hsv_v_entry.delete("1.0", tk.END)
        lr0_entry.delete("1.0", tk.END)
        lrf_entry.delete("1.0", tk.END)
        b_entry.configure(bg='white')
        e_entry.configure(bg='white')
        hsv_s_entry.configure(bg='white')
        hsv_v_entry.configure(bg='white')
        lr0_entry.configure(bg='white')
        lrf_entry.configure(bg='white')
        canvas1.delete("all")
        canvas2.delete("all")
        canvas1.config(width=max_canvas_width, height=max_canvas_height)
        canvas2.config(width=max_canvas_width, height=max_canvas_height)
        parar_video()

    if op == 2:
        # Exibe todos os campos para configuração do treinamento
        e_label.place(x=400, y=680, anchor=CENTER)
        e_entry.place(x=450, y=680, anchor=CENTER)
        b_label.place(x=400, y=720, anchor=CENTER)
        b_entry.place(x=450, y=720, anchor=CENTER)
        hsv_s_label.place(x=500, y=680, anchor=CENTER)
        hsv_s_entry.place(x=550, y=680, anchor=CENTER)
        hsv_v_label.place(x=500, y=720, anchor=CENTER)
        hsv_v_entry.place(x=550, y=720, anchor=CENTER)
        lr0_label.place(x=600, y=680, anchor=CENTER)
        lr0_entry.place(x=650, y=680, anchor=CENTER)
        lrf_label.place(x=600, y=720, anchor=CENTER)
        lrf_entry.place(x=650, y=720, anchor=CENTER)
        canvas1.delete("all")
        canvas2.delete("all")
        canvas1.config(width=max_canvas_width, height=max_canvas_height)
        canvas2.config(width=max_canvas_width, height=max_canvas_height)
        parar_video()

    if op == 3:
        # Caso 3: limpa tudo (sem mostrar campos)
        b_entry.delete("1.0", tk.END)
        e_entry.delete("1.0", tk.END)
        hsv_s_entry.delete("1.0", tk.END)
        hsv_v_entry.delete("1.0", tk.END)
        lr0_entry.delete("1.0", tk.END)
        lrf_entry.delete("1.0", tk.END)
        b_entry.configure(bg='white')
        e_entry.configure(bg='white')
        hsv_s_entry.configure(bg='white')
        hsv_v_entry.configure(bg='white')
        lr0_entry.configure(bg='white')
        lrf_entry.configure(bg='white')
        canvas1.delete("all")
        canvas2.delete("all")
        canvas1.config(width=max_canvas_width, height=max_canvas_height)
        canvas2.config(width=max_canvas_width, height=max_canvas_height)
        parar_video()

    if op == 4:
        # Mostra botão de escolher imagem
        b_entry.delete("1.0", tk.END)
        e_entry.delete("1.0", tk.END)
        hsv_s_entry.delete("1.0", tk.END)
        hsv_v_entry.delete("1.0", tk.END)
        lr0_entry.delete("1.0", tk.END)
        lrf_entry.delete("1.0", tk.END)
        b_entry.configure(bg='white')
        e_entry.configure(bg='white')
        hsv_s_entry.configure(bg='white')
        hsv_v_entry.configure(bg='white')
        lr0_entry.configure(bg='white')
        lrf_entry.configure(bg='white')
        btn1.place(x=370, y=660, anchor=NW)
        parar_video()

    if op == 5:
        # Similar ao caso 3: apenas limpa e reseta tudo
        b_entry.delete("1.0", tk.END)
        e_entry.delete("1.0", tk.END)
        hsv_s_entry.delete("1.0", tk.END)
        hsv_v_entry.delete("1.0", tk.END)
        lr0_entry.delete("1.0", tk.END)
        lrf_entry.delete("1.0", tk.END)
        b_entry.configure(bg='white')
        e_entry.configure(bg='white')
        hsv_s_entry.configure(bg='white')
        hsv_v_entry.configure(bg='white')
        lr0_entry.configure(bg='white')
        lrf_entry.configure(bg='white')
        canvas1.delete("all")
        canvas2.delete("all")
        canvas1.config(width=max_canvas_width, height=max_canvas_height)
        canvas2.config(width=max_canvas_width, height=max_canvas_height)

# Função que executa ação conforme item selecionado e valida os campos se necessário
def esc_select():
    global img_pil

    op = listbox.curselection()
    if not op:
        return
    op = op[0]

    if op == 0:
        # Cria o ficheiro data.yaml
        criar_data_yaml()

    if op == 1:
        # Prepara o dataset para treino
        preparadataset(base_path)
        print('Preparadataset realizado')

    if op == 2:
        # Valida os campos de input e inicia o treino do modelo
        try:
            b_entry.configure(bg='white')
            e_entry.configure(bg='white')
            hsv_s_entry.configure(bg='white')
            hsv_v_entry.configure(bg='white')
            lr0_entry.configure(bg='white')
            lrf_entry.configure(bg='white')

            b_value = b_entry.get("1.0", "end-1c")
            e_value = e_entry.get("1.0", "end-1c")
            hsv_s_value = hsv_s_entry.get("1.0", "end-1c")
            hsv_v_value = hsv_v_entry.get("1.0", "end-1c")
            lr0_value = lr0_entry.get("1.0", "end-1c")
            lrf_value = lrf_entry.get("1.0", "end-1c")

            # Funções auxiliares para validação
            def is_empty(value):
                return value.strip() == ""

            def is_int(value):
                try:
                    int(value)
                    return True
                except ValueError:
                    return False

            def is_float(value):
                try:
                    float(value)
                    return True
                except ValueError:
                    return False

            erro = False

            # Validação individual dos campos
            if is_empty(e_value) or not is_int(e_value) or not (30 <= int(e_value) <= 150):
                e_entry.configure(bg='red')
                erro = True
            if is_empty(b_value) or not is_int(b_value) or int(b_value) not in {2, 4, 8, 16, 32, 64}:
                b_entry.configure(bg='red')
                erro = True
            if is_empty(hsv_s_value) or not is_float(hsv_s_value) or not (0.0 <= float(hsv_s_value) <= 1.0):
                hsv_s_entry.configure(bg='red')
                erro = True
            if is_empty(hsv_v_value) or not is_float(hsv_v_value) or not (0.0 <= float(hsv_v_value) <= 1.0):
                hsv_v_entry.configure(bg='red')
                erro = True
            if is_empty(lr0_value) or not is_float(lr0_value) or not (0.000001 <= float(lr0_value) <= 0.1):
                lr0_entry.configure(bg='red')
                erro = True
            if is_empty(lrf_value) or not is_float(lrf_value) or not (0.000001 <= float(lrf_value) <= 0.1):
                lrf_entry.configure(bg='red')
                erro = True

            if erro:
                raise ValueError

        except ValueError:
            show_error_message("Considere as seguintes condições: Epochs - entre 30 e 150; Batch - entre 2 e 64; hsv_s e hsv_v - entre 0.0 e 1.0; lr0 e lrf - entre 0.000001 e 0.1")
            return

        yaml_path = verificar_ficheiro_yaml()
        if yaml_path:
            train_dir, val_dir = obter_diretorios_dados()
            if train_dir and val_dir:
                atualizar_config()
                mostrar_mensagem_constante("A iniciar subprocesso para treino...")
                win.after(2000)
                mostrar_mensagem_constante("A treinar... Por favor, aguarde.")
                threading.Thread(target=correr_treino).start()

    if op == 3:
        # Mostra resultados do treino
        mostrar_resultados()

    if op == 4:
        # Classifica uma imagem com o modelo
        if not os.path.exists('best.pt'):
            show_error_message("Erro: Modelo 'best.pt' não encontrado.")
            return
        if not filename:
            show_error_message("Erro: Nenhuma imagem carregada para classificar. Por favor escolha uma imagem para classificar.")
            return
        testar_imagem_com_modelo()

    if op == 5:
        # Detecta objetos em vídeo (com câmera)
        if not os.path.exists('best.pt'):
            show_error_message("Erro: Modelo 'best.pt' não encontrado.")
            return
        detetar_video_camera()

# Definição do Layout
btn1 = Button(win, text="Escolher Imagem", command=esc_img)

canvas1 = Canvas(win, width=700, height=500, bd=0, highlightthickness=0, bg="white") 
canvas1.place(x=15, y=340, anchor=W)

canvas2 = Canvas(win, width=700, height=500, bd=0, highlightthickness=0, bg="white")
canvas2.place(x=775, y=340, anchor=W)

Label(win, text="Imagem Original", font=("Arial", 12)).place(x=15, y=50, anchor=NW)
Label(win, text="Imagem Classificada", font=("Arial", 12)).place(x=775, y=50, anchor=NW)
Label(win, text="Opções", font=("Arial", 12)).place(x=15, y=620, anchor=SW)

listbox = Listbox(win, height=6, width=35, exportselection=False, font=("Helvetica", 11))
listbox.insert(1, "Criação do data.yaml")
listbox.insert(2, "Preparar dataset")
listbox.insert(3, "Treinar modelo")
listbox.insert(4, "Mostrar resultados")
listbox.insert(5, "Classificar imagem")
listbox.insert(6, "Classificar em vídeo")
listbox.place(x=15, y=735, anchor=SW)
listbox.bind('<ButtonRelease-1>', lambda event: select_listbox())
listbox.select_set(0)

# Definição de alterar Epochs
e_label = Label(win, text="Epochs:", font=("Arial", 10))
e_entry = Text(win, width=5, height=1)

# Definição de alterar Batch
b_label = Label(win, text="Batch:", font=("Arial", 10))
b_entry = Text(win, width=5, height=1)

# Definição de alterar HSV_S 
hsv_s_label = Label(win, text="hsv_s:", font=("Arial", 10))
hsv_s_entry = Text(win, width=5, height=1)

# Definição de alterar HSV_V 
hsv_v_label = Label(win, text="hsv_v:", font=("Arial", 10))
hsv_v_entry = Text(win, width=5, height=1)
 
# Definição de alterar lr0 e lrf
lr0_label = Label(win, text="lr0:", font=("Arial", 10))
lr0_entry = Text(win, width=10, height=1)

lrf_label = Label(win, text="lrf:", font=("Arial", 10))
lrf_entry = Text(win, width=10, height=1)

# Associar função para reset do fundo ao clicar
for entry in [e_entry, b_entry, hsv_s_entry, hsv_v_entry, lr0_entry, lrf_entry]:
    entry.bind("<FocusIn>", reset_bg)

# Botão Aplicar
btn2 = Button(win, text="Aplicar", command=esc_select)
btn2.place(x=15, y=770, anchor=SW)

win.mainloop()


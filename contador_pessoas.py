import cv2
from ultralytics import YOLO


def main():
    # Carregar modelo YOLOv8 (detecta pessoas - classe 0 do COCO)
    modelo = YOLO("yolov8n.pt")

    # Para testes, usar o vídeo runners.mp4
    # Para câmera USB ao vivo, substituir por 0
    fonte_video = "runners2.mp4"
    # fonte_video = 0  # Câmera USB

    cap = cv2.VideoCapture(fonte_video)

    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir a fonte de vídeo '{fonte_video}'")
        return

    # Dimensões originais do vídeo
    largura_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Janela redimensionável com 50% do tamanho original
    largura_janela = largura_orig // 2
    altura_janela = altura_orig // 2
    cv2.namedWindow("Contador de Pessoas - COGNICORE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Contador de Pessoas - COGNICORE", largura_janela, altura_janela)

    print("Pressione 'q' ou ESC para sair.")

    # Tracking simples por centroides
    centroides_anterior = {}  # id -> (cx, cy, frames_sem_deteccao)
    proximo_id = 0
    total_pessoas = 0
    caixas_salvas = []
    num_frame = 0
    PULAR_FRAMES = 3  # Processar detecção a cada N frames
    DIST_MAX = 100  # Distância máxima para associar mesmo objeto

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            num_frame = 0
            continue

        num_frame += 1

        # Rodar detecção a cada N frames
        if num_frame % PULAR_FRAMES == 0:
            # Redimensionar frame para inferência rápida
            frame_pequeno = cv2.resize(frame, (416, 234))
            escala_x = largura_orig / 416
            escala_y = altura_orig / 234

            resultados = modelo(frame_pequeno, verbose=False, conf=0.25,
                                classes=[0])

            caixas_salvas = []
            centroides_atual = []

            for resultado in resultados:
                boxes = resultado.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1 = int(x1 * escala_x)
                    y1 = int(y1 * escala_y)
                    x2 = int(x2 * escala_x)
                    y2 = int(y2 * escala_y)

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    caixas_salvas.append((x1, y1, x2, y2))
                    centroides_atual.append((cx, cy))

            # Tracking simples por proximidade
            novos_centroides = {}
            ids_usados = set()

            for (cx, cy) in centroides_atual:
                melhor_id = None
                melhor_dist = float('inf')

                for obj_id, (prev_cx, prev_cy, _) in centroides_anterior.items():
                    if obj_id in ids_usados:
                        continue
                    dist = ((cx - prev_cx)**2 + (cy - prev_cy)**2) ** 0.5
                    if dist < melhor_dist and dist < DIST_MAX:
                        melhor_dist = dist
                        melhor_id = obj_id

                if melhor_id is not None:
                    ids_usados.add(melhor_id)
                    novos_centroides[melhor_id] = (cx, cy, 0)
                else:
                    # Nova pessoa detectada
                    novos_centroides[proximo_id] = (cx, cy, 0)
                    proximo_id += 1
                    total_pessoas += 1

            # Mantem rastros por alguns frames para evitar recontagem apos uma
            # falha pontual na deteccao.
            for obj_id, (prev_cx, prev_cy, frames_sem_deteccao) in centroides_anterior.items():
                if obj_id not in novos_centroides and frames_sem_deteccao < 30:
                    novos_centroides[obj_id] = (
                        prev_cx, prev_cy, frames_sem_deteccao + 1)

            centroides_anterior = novos_centroides

        # Desenhar retângulos das detecções
        for (x1, y1, x2, y2) in caixas_salvas:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Desenhar painel de contagem no topo da tela
        texto = f"PESSOAS: {total_pessoas}"
        tamanho_texto = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]

        margem = 10
        x_rect = 10
        y_rect = 10
        largura_rect = tamanho_texto[0] + margem * 2
        altura_rect = tamanho_texto[1] + margem * 2 + 10

        # Retângulo verde de fundo
        cv2.rectangle(frame, (x_rect, y_rect),
                      (x_rect + largura_rect, y_rect + altura_rect),
                      (0, 200, 0), -1)

        # Borda do retângulo
        cv2.rectangle(frame, (x_rect, y_rect),
                      (x_rect + largura_rect, y_rect + altura_rect),
                      (0, 255, 0), 2)

        # Texto branco sobre o retângulo verde
        cv2.putText(frame, texto,
                    (x_rect + margem, y_rect + tamanho_texto[1] + margem + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # Redimensionar para exibição (50% do original)
        frame_display = cv2.resize(frame, (largura_janela, altura_janela))

        # Mostrar o frame
        cv2.imshow("Contador de Pessoas - COGNICORE", frame_display)

        # Sair com 'q' ou ESC
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q') or tecla == ord('Q') or tecla == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    print(f"\nTotal de pessoas contadas: {total_pessoas}")


if __name__ == "__main__":
    main()

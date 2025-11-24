# ------------------------------------
# ไฟล์: app.py
# ------------------------------------
import gradio as gr
import cv2
from ultralytics import YOLO
from voice_guidance import speak_guidance  # Import ฟังก์ชันพูด
import threading
import time

# -------------------------------------------------------------------
# (สำคัญ!) แก้ไข Path นี้ให้ตรงกับไฟล์ best.pt ที่คุณเทรนได้
# -------------------------------------------------------------------
MODEL_PATH = 'runs/detect/yolo12m_final/weights/best.pt' # << แก้ไขตรงนี้
# -------------------------------------------------------------------

# 1. โหลดโมเดล AI ที่เทรนเสร็จแล้ว
try:
    print(f"กำลังโหลดโมเดลจาก: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("โหลดโมเดลสำเร็จ")
except Exception as e:
    print(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    print("กรุณาตรวจสอบว่า Path ของโมเดลถูกต้องหรือไม่")
    exit()

# การตั้งค่าเพิ่มเติมสำหรับระบบเสียง
SPEECH_CONF_THRESHOLD = 0.4          # conf ขั้นต่ำที่จะพิจารณาพูด (ลดเพื่อให้พูดง่ายขึ้น)
SUSTAINED_FRAME_THRESHOLD = 2        # จำนวนเฟรมต่อเนื่องก่อนพูด (ลดความเข้มงวด)
ANNOUNCE_COOLDOWN_SECONDS = 6        # เวลาระหว่างการพูดซ้ำคลาสเดิม

# ตัวแปรสถานะสำหรับระบบเสียง
last_detected_class_for_speech = -1
current_streak_class = -1
current_streak_length = 0
last_announced_class = -1
last_announced_time = 0.0

def run_speech_in_background():
    """
    ฟังก์ชันนี้จะรันใน Thread แยก
    เพื่อเรียกใช้ speak_guidance โดยไม่ทำให้วิดีโอค้าง
    """
    global last_detected_class_for_speech
    while True:
        if last_detected_class_for_speech != -1:
            # เก็บค่าไว้ก่อน แล้วรีเซ็ตทันที
            class_to_speak = last_detected_class_for_speech
            last_detected_class_for_speech = -1
            
            # เรียกใช้ฟังก์ชันพูด (ซึ่งมี Debounce ของตัวเอง)
            speak_guidance(class_to_speak)
        # ป้องกัน busy loop
        time.sleep(0.05)

def process_frame(frame):
    """
    ฟังก์ชันหลักที่ Gradio จะเรียกใช้สำหรับทุกเฟรมจาก Webcam
    """
    global last_detected_class_for_speech, current_streak_class, current_streak_length
    global last_announced_class, last_announced_time
    
    # 1. พลิกเฟรม (กล้อง Webcam มักจะกลับด้าน)
    frame = cv2.flip(frame, 1)
    # แปลงเป็น BGR สำหรับโมเดล (Gradio ป้อน RGB)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # 2. สั่งให้โมเดลตรวจจับวัตถุในเฟรม
    results = model(frame_bgr, conf=0.25, imgsz=640, verbose=False, max_det=300)
    
    # 3. วาดกรอบและชื่อคลาสลงบนภาพ (ฟังก์ชัน .plot() ของ ultralytics)
    annotated_bgr = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    try:
        print("boxes:", len(results[0].boxes))
    except Exception:
        pass
    
    # 4. ตรวจสอบว่าเจออะไรหรือไม่
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        detected_class_tensor = boxes.cls[0]
        detected_class = int(detected_class_tensor.item())
        detected_conf = float(boxes.conf[0].item()) if boxes.conf is not None else 0.0

        if detected_conf >= SPEECH_CONF_THRESHOLD:
            if detected_class == current_streak_class:
                current_streak_length += 1
            else:
                current_streak_class = detected_class
                current_streak_length = 1

            now = time.time()
            should_trigger_speech = (
                current_streak_length >= SUSTAINED_FRAME_THRESHOLD and
                (
                    detected_class != last_announced_class or
                    now - last_announced_time >= ANNOUNCE_COOLDOWN_SECONDS
                )
            )

            if should_trigger_speech:
                # ส่งคำสั่งให้พูด (เสียงจะพูดจนเสร็จแม้ไม่มี detection ต่อ)
                last_detected_class_for_speech = detected_class
                last_announced_class = detected_class
                last_announced_time = now
                try:
                    print(f"[SPEECH] trigger class={detected_class} conf={detected_conf:.2f}")
                except Exception:
                    pass
        else:
            # conf ต่ำเกินไป - reset streak แต่ไม่หยุดเสียงที่กำลังพูดอยู่
            current_streak_class = -1
            current_streak_length = 0
    else:
        # ไม่เจอวัตถุ - reset streak แต่ไม่หยุดเสียงที่กำลังพูดอยู่
        # หมายเหตุ: เสียงที่ส่งเข้า queue แล้วจะพูดจนเสร็จ ไม่ว่าจะมี detection ต่อหรือไม่
        current_streak_class = -1
        current_streak_length = 0

    # คืนค่าภาพที่มีกรอบวาดแล้ว กลับไปแสดงที่หน้าเว็บ
    return annotated_frame

# --- สร้าง Gradio Interface ---
def main():
    print("กำลังสร้าง Gradio Interface...")
    
    # เริ่ม Thread สำหรับการพูดแยกต่างหาก
    speech_thread = threading.Thread(target=run_speech_in_background, daemon=True)
    speech_thread.start()

    # สร้างหน้าเว็บ
    iface = gr.Interface(
        fn=process_frame,
        inputs=gr.Image(
            type="numpy", 
            sources=["webcam"],
            streaming=True,
            label="จ่อขยะที่กล้องนี้"
        ),
        outputs=gr.Image(
            type="numpy", 
            label="ผลการตรวจจับ"
        ),
        live=True, # ทำให้เป็น Real-time
        title="🤖 ระบบคัดแยกขยะอัจฉริยะ (AI Waste Sorter)",
        description="โครงงานโดย: พงศภัค, กฤติน, ภูริชทัต (ใช้ YOLOv12)"
    )
    
    # 8. รันแอป
    print("Interface พร้อมใช้งาน. เปิดในเบราว์เซอร์ของคุณ...")
    iface.launch(share=False) # share=True ถ้าต้องการส่งลิงก์ให้คนอื่นดู

if __name__ == '__main__':
    main()
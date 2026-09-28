import os
import cv2
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from collections import defaultdict
from dataclasses import dataclass

from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


@dataclass 
class PredictionConfig:
    model_path_best_pt: str = os.path.join('model', 'best_20.pt')
    conf_threshold: float = 0.65
    output_dir: str = 'output'


class Prediction:
    def __init__(self, config: PredictionConfig = PredictionConfig()):
        self.config = config
        self.model = YOLO(self.config.model_path_best_pt)

        # Standardized weights mapping (IDs 0–5 & string names)
        self.class_weights = {
            0: 5.0, 'pothole': 5.0,
            1: 3.0, 'crack': 3.0,
            2: 2.5, 'debris': 2.5,
            3: 2.0, 'manhole': 2.0,
            4: 0.5, 'speed breaker': 0.5,
            5: 1.0, 'road barricade': 1.0
        }
        
        self.class_id_to_name = {
            0: "Pothole",
            1: "Crack",
            2: "Debris",
            3: "Manhole",
            4: "Speed Breaker",
            5: "Road Barricade"
        }

    # ==========================================
    # 5. SEVERITY METER (UNIFIED FOR ALL MODES)
    # ==========================================
    def calculate_severity(self, counts: dict, max_score: float = 10.0):
        """
        Unified severity score calculator.
        Accepts dicts with either integer Class IDs or String Names as keys.
        Returns: (score, priority_label, remark)
        """
        if not counts:
            return 0.0, "LOW", "GOOD - Road is in clean condition with no detected defects."

        total_score = 0.0
        for cls_key, count in counts.items():
            if isinstance(cls_key, str):
                cls_key = cls_key.lower().strip()
            weight = self.class_weights.get(cls_key, 1.0)
            total_score += count * weight

        # Cap score option for single image/video if max_score is provided
        score = round(min(max_score, total_score) if max_score else total_score, 2)

        if score >= 25.0 or (max_score == 10.0 and score >= 8.5):
            priority = "CRITICAL"
            remark = "CRITICAL - Severe road damage. Requires immediate inspection and repair."
        elif score >= 10.0 or (max_score == 10.0 and score >= 6.0):
            priority = "HIGH"
            remark = "POOR - High density of hazards detected. Prompt repairs required."
        elif score >= 3.0:
            priority = "MEDIUM"
            remark = "FAIR - Moderate damage detected. Road needs routine maintenance."
        else:
            priority = "LOW"
            remark = "GOOD - Minor or low-hazard defects present."

        return score, priority, remark

    # ==========================================
    # MATH & INTERPOLATION HELPERS
    # ==========================================
    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates distance between two lat/lon points in meters."""
        r = 6371000.0
        dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
        a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
        return 2 * r * asin(sqrt(a))

    def interpolate_gps(self, df_gps: pd.DataFrame, t_video: float):
        """Linearly interpolates lat, lon, and speed for a given timestamp."""
        if t_video <= df_gps['timestamp_sec'].iloc[0]:
            r0 = df_gps.iloc[0]
            return float(r0['latitude']), float(r0['longitude']), float(r0['speed_kmh'])
        if t_video >= df_gps['timestamp_sec'].iloc[-1]:
            r_last = df_gps.iloc[-1]
            return float(r_last['latitude']), float(r_last['longitude']), float(r_last['speed_kmh'])

        idx_b = (df_gps['timestamp_sec'] > t_video).idxmax()
        idx_a = idx_b - 1

        ta, tb = df_gps.loc[idx_a, 'timestamp_sec'], df_gps.loc[idx_b, 'timestamp_sec']
        alpha = (t_video - ta) / (tb - ta) if tb != ta else 0.0

        lat = df_gps.loc[idx_a, 'latitude'] + alpha * (df_gps.loc[idx_b, 'latitude'] - df_gps.loc[idx_a, 'latitude'])
        lon = df_gps.loc[idx_a, 'longitude'] + alpha * (df_gps.loc[idx_b, 'longitude'] - df_gps.loc[idx_a, 'longitude'])
        speed = df_gps.loc[idx_a, 'speed_kmh'] + alpha * (df_gps.loc[idx_b, 'speed_kmh'] - df_gps.loc[idx_a, 'speed_kmh'])

        return float(lat), float(lon), float(speed)

    # ==========================================
    # 1. IMAGE ANALYZER
    # ==========================================
    def image_classifier(self, path_to_image: str, generate_pdf: bool = True):
        """Processes a single image, evaluates severity score, and optionally exports PDF report."""
        image_name = os.path.splitext(os.path.basename(path_to_image))[0]
        os.makedirs(self.config.output_dir, exist_ok=True)

        results = self.model.predict(
            source=path_to_image,
            conf=self.config.conf_threshold,
            save=True,
            project=self.config.output_dir,
            name=f"{image_name}_op",
            exist_ok=True
        )

        boxes = results[0].boxes
        counts = defaultdict(int)

        if boxes is not None and len(boxes) > 0:
            clss = boxes.cls.cpu().numpy().astype(int)
            for cls_id in clss:
                class_name = self.model.names[cls_id]
                counts[class_name] += 1

        score, priority, remark = self.calculate_severity(counts, max_score=10.0)

        result_payload = {
            "image_path": path_to_image,
            "counts": dict(counts),
            "severity_score": score,
            "priority": priority,
            "remark": remark
        }

        if generate_pdf:
            pdf_path = os.path.join(self.config.output_dir, f"{image_name}_report.pdf")
            self.generate_pdf_report(pdf_path, report_type="image", data=result_payload)
            result_payload["pdf_path"] = pdf_path

        return result_payload

    # ==========================================
    # 2. VIDEO ANALYZER (NO GPS)
    # ==========================================
    def video_classifier(self, video_path: str, display_preview: bool = False, generate_pdf: bool = True):
        """Runs tracking on standalone video using ByteTrack and calculates route severity score."""
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(self.config.output_dir, exist_ok=True)

        output_video_path = os.path.join(self.config.output_dir, f"{base_name}_detected.mp4")

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Error opening video at {video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        unique_ids = defaultdict(set)

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=self.config.conf_threshold, verbose=False)

            if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                ids = boxes.id.cpu().numpy().astype(int)
                clss = boxes.cls.cpu().numpy().astype(int)
                for obj_id, cls_id in zip(ids, clss):
                    class_name = self.model.names[cls_id]
                    unique_ids[class_name].add(obj_id)

                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame

            out_writer.write(annotated_frame)

            if display_preview:
                cv2.imshow('YOLO Video Tracking', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        capture.release()
        out_writer.release()
        if display_preview:
            cv2.destroyAllWindows()

        final_counts = {cls_name: len(ids) for cls_name, ids in unique_ids.items()}
        score, priority, remark = self.calculate_severity(final_counts, max_score=10.0)

        result_payload = {
            "video_path": video_path,
            "output_video_path": output_video_path,
            "resolution": f"{width}x{height}",
            "fps": fps,
            "counts": final_counts,
            "severity_score": score,
            "priority": priority,
            "remark": remark
        }

        if generate_pdf:
            pdf_path = os.path.join(self.config.output_dir, f"{base_name}_report.pdf")
            self.generate_pdf_report(pdf_path, report_type="video", data=result_payload)
            result_payload["pdf_path"] = pdf_path

        return result_payload

    # ==========================================
    # 3. VIDEO + GPS ANALYZER (500M CHAINAGE)
    # ==========================================
    def video_gps_classifier(self, video_path: str, gps_csv_path: str, generate_pdf: bool = True):
        """Synchronizes video with 1Hz GPS, partitions 500m segments, and scores each segment."""
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        os.makedirs(self.config.output_dir, exist_ok=True)

        output_video_path = os.path.join(self.config.output_dir, f"{base_name}_gps_detected.mp4")
        df_gps = pd.read_csv(gps_csv_path).sort_values("timestamp_sec").reset_index(drop=True)

        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"Error opening video at {video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        active_set_num = 1
        d_accum = 0.0
        total_route_distance = 0.0
        prev_coords = None
        global_seen_track_ids = set()

        current_set_defects = {c: [] for c in range(6)}
        set_start_coords = self.interpolate_gps(df_gps, 0.0)[:2]
        segments_summary = []

        frame_idx = 0

        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            t_curr = frame_idx / fps
            curr_lat, curr_lon, curr_speed = self.interpolate_gps(df_gps, t_curr)

            # Stationary filter & step calculation
            if prev_coords is not None and curr_speed >= 3.0:
                dist_step = self.haversine(prev_coords[0], prev_coords[1], curr_lat, curr_lon)
                if dist_step < 35.0:
                    d_accum += dist_step
                    total_route_distance += dist_step

            prev_coords = (curr_lat, curr_lon)

            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=self.config.conf_threshold, verbose=False)

            if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                ids = boxes.id.cpu().numpy().astype(int)
                clss = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy().astype(float)

                for obj_id, cls_id, conf in zip(ids, clss, confs):
                    if obj_id not in global_seen_track_ids:
                        global_seen_track_ids.add(obj_id)
                        current_set_defects[cls_id].append({
                            "track_id": obj_id,
                            "lat": curr_lat,
                            "lon": curr_lon,
                            "time": t_curr,
                            "conf": conf
                        })

                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame.copy()

            # 500-Meter Chainage Boundary Trigger
            if d_accum >= 500.0:
                counts = {cls_id: len(items) for cls_id, items in current_set_defects.items()}
                score, priority, _ = self.calculate_severity(counts, max_score=None)

                segments_summary.append({
                    "set_number": active_set_num,
                    "start_coords": set_start_coords,
                    "end_coords": (curr_lat, curr_lon),
                    "counts": counts,
                    "score": score,
                    "priority": priority,
                    "defects": current_set_defects
                })

                active_set_num += 1
                d_accum = 0.0
                set_start_coords = (curr_lat, curr_lon)
                current_set_defects = {c: [] for c in range(6)}

            # Render Telemetry HUD
            cv2.rectangle(annotated_frame, (10, 10), (380, 110), (0, 0, 0), -1)
            cv2.putText(annotated_frame, f"Speed: {curr_speed:.1f} km/h", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Set #: {active_set_num} | Chainage: {d_accum:.1f}m", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Lat: {curr_lat:.6f} Lon: {curr_lon:.6f}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            out_writer.write(annotated_frame)
            frame_idx += 1

        # Flush Partial Final Segment
        if d_accum > 0 or frame_idx > 0:
            counts = {cls_id: len(items) for cls_id, items in current_set_defects.items()}
            score, priority, _ = self.calculate_severity(counts, max_score=None)

            segments_summary.append({
                "set_number": active_set_num,
                "start_coords": set_start_coords,
                "end_coords": prev_coords or set_start_coords,
                "counts": counts,
                "score": score,
                "priority": priority,
                "defects": current_set_defects
            })

        capture.release()
        out_writer.release()

        result_payload = {
            "video_path": video_path,
            "output_video_path": output_video_path,
            "total_distance_km": round(total_route_distance / 1000.0, 2),
            "total_segments": active_set_num,
            "segments": segments_summary
        }

        if generate_pdf:
            pdf_path = os.path.join(self.config.output_dir, f"{base_name}_gps_report.pdf")
            self.generate_pdf_report(pdf_path, report_type="video_gps", data=result_payload)
            result_payload["pdf_path"] = pdf_path

        return result_payload

    # ==========================================
    # 4. PDF MAKER (UNIFIED REPORT GENERATOR)
    # ==========================================
    def generate_pdf_report(self, pdf_path: str, report_type: str, data: dict):
        """Constructs PDF summary reports for Image, Video, or Video+GPS runs."""
        doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, textColor=colors.HexColor('#1A2B4C'), spaceAfter=10)
        normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#333333'))
        header_style = ParagraphStyle('HeaderStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10, textColor=colors.white)

        elements = []

        if report_type in ["image", "video"]:
            title_text = "Image Detection Report" if report_type == "image" else "Video Detection Report"
            file_key = "image_path" if report_type == "image" else "video_path"

            elements.extend([
                Paragraph(f"YOLO {title_text}", title_style),
                Paragraph(f"<b>Input File:</b> {os.path.basename(data[file_key])}", normal_style),
                Spacer(1, 8),
                Paragraph(f"<b>Damage Severity Score:</b> {data['severity_score']} / 10.0 ({data['priority']})", normal_style),
                Paragraph(f"<b>Remark:</b> {data['remark']}", normal_style),
                Spacer(1, 15)
            ])

            table_data = [[Paragraph("<b>Class Name</b>", header_style), Paragraph("<b>Count</b>", header_style)]]
            counts = data.get("counts", {})
            if counts:
                for cls_name, count in sorted(counts.items()):
                    table_data.append([Paragraph(str(cls_name).upper(), normal_style), Paragraph(str(count), normal_style)])
            else:
                table_data.append([Paragraph("No defects detected", normal_style), Paragraph("0", normal_style)])

            table = Table(table_data, colWidths=[250, 250])

        elif report_type == "video_gps":
            elements.extend([
                Paragraph("Spatial Road Audit Report (Video + GPS)", title_style),
                Paragraph(f"<b>Input File:</b> {os.path.basename(data['video_path'])}<br/>"
                          f"<b>Total Distance:</b> {data['total_distance_km']} km | <b>500m Sets:</b> {data['total_segments']}", normal_style),
                Spacer(1, 15)
            ])

            table_data = [[
                Paragraph("<b>Set #</b>", header_style),
                Paragraph("<b>Pothole</b>", header_style),
                Paragraph("<b>Crack</b>", header_style),
                Paragraph("<b>Debris</b>", header_style),
                Paragraph("<b>Manhole</b>", header_style),
                Paragraph("<b>Barricade</b>", header_style),
                Paragraph("<b>Score</b>", header_style),
                Paragraph("<b>Priority</b>", header_style)
            ]]

            for seg in data.get("segments", []):
                c = seg["counts"]
                table_data.append([
                    Paragraph(f"Set #{seg['set_number']}", normal_style),
                    Paragraph(str(c.get(0, 0)), normal_style),
                    Paragraph(str(c.get(1, 0)), normal_style),
                    Paragraph(str(c.get(2, 0)), normal_style),
                    Paragraph(str(c.get(3, 0)), normal_style),
                    Paragraph(str(c.get(5, 0)), normal_style),
                    Paragraph(f"{seg['score']:.1f}", normal_style),
                    Paragraph(f"<b>{seg['priority']}</b>", normal_style)
                ])

            table = Table(table_data, colWidths=[45, 55, 50, 50, 60, 60, 50, 70])

        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A2B4C')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DDDDDD')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#1A2B4C')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))

        elements.append(table)
        doc.build(elements)


if __name__ == "__main__":
    predictor = Prediction()
    # 1. Image Example:
    # img_res = predictor.image_classifier('test/sample.jpg')

    # 2. Standalone Video Example:
    # vid_res = predictor.video_classifier('test/sample.mp4')

    # 3. Video + GPS Example:
    # gps_res = predictor.video_gps_classifier('test/sample.mp4', 'test/route_telemetry.csv')
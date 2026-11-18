import os
from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, ForeignKey, text
from sqlalchemy.orm import relationship
from src.database.engine import ENGINE, BASE, SESSION_LOCAL

class RunModel(BASE):
    __tablename__ = 'runs'
    
    run_id = Column(Integer, primary_key=True, autoincrement=True)
    mode = Column(String(20), nullable=False)      # 'image', 'video', or 'video_gps'
    input_path = Column(String(255), nullable=False)
    output_path = Column(String(255), nullable=True)
    report_path = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    
    detections = relationship("DetectionModel", back_populates="run", cascade="all, delete-orphan")


class DetectionModel(BASE):
    __tablename__ = 'detections'
    
    det_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey('runs.run_id', ondelete='CASCADE'), nullable=False)
    track_id = Column(Integer, nullable=True)       # ByteTrack ID, NULL for single images
    class_name = Column(String(30), nullable=False) # pothole, crack, debris, etc.
    confidence = Column(Float, nullable=True)
    frame_no = Column(Integer, nullable=True)       # NULL for single images
    
    run = relationship("RunModel", back_populates="detections")


def init_db():
    """1. Creates the database tables if they do not exist."""
    BASE.metadata.create_all(bind=ENGINE)
    print("SUCCESS: Database tables initialized successfully via SQLAlchemy.")


def create_run(mode, input_path):
    """2. Creates a new run record and returns the new run_id."""
    session = SESSION_LOCAL()
    try:
        new_run = RunModel(mode=mode, input_path=input_path)
        session.add(new_run)
        session.commit()
        session.refresh(new_run)
        return new_run.run_id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def add_detections(run_id, detections):
    """3. Bulk insert detections using a list of dictionaries."""
    if not detections:
        return
    session = SESSION_LOCAL()
    try:
        detection_objects = [
            DetectionModel(
                run_id=run_id,
                track_id=d.get('track_id'),
                class_name=d['class_name'],
                confidence=d.get('confidence'),
                frame_no=d.get('frame_no')
            )
            for d in detections
        ]
        session.bulk_save_objects(detection_objects)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def update_run_outputs(run_id, output_path, report_path):
    """4. Fills in the result paths after processing finishes."""
    session = SESSION_LOCAL()
    try:
        run = session.get(RunModel, run_id)
        if run:
            run.output_path = output_path
            run.report_path = report_path
            session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_class_counts(run_id):
    """5. Returns defect counts per class. 
       For video/video_gps, counts unique track_id per class (COUNT DISTINCT).
       For images where track_id is NULL, counts total occurrences (COUNT(*)).
    """
    session = SESSION_LOCAL()
    try:
        run = session.get(RunModel, run_id)
        if not run:
            return {}
        
        detections = run.detections
        counts = {}
        
        if run.mode == 'image':
            for d in detections:
                counts[d.class_name] = counts.get(d.class_name, 0) + 1
        else:
            # Video modes: Count unique track_id per class
            unique_tracks_per_class = {}
            for d in detections:
                if d.track_id is not None:
                    if d.class_name not in unique_tracks_per_class:
                        unique_tracks_per_class[d.class_name] = set()
                    unique_tracks_per_class[d.class_name].add(d.track_id)
            for cls_name, tracks in unique_tracks_per_class.items():
                counts[cls_name] = len(tracks)
                
        return counts
    finally:
        session.close()


def get_severity_score(run_id):
    """6. Sums (defect count x hazard weight) using a Python dictionary."""
    hazard_weights = {
        'pothole': 5.0,
        'crack': 3.0,
        'debris': 2.5,
        'manhole': 2.0,
        'road barricade': 1.0,
        'speed breaker': 0.5
    }
    counts = get_class_counts(run_id)
    total_score = 0.0
    for class_name, count in counts.items():
        weight = hazard_weights.get(class_name.lower().strip(), 1.0)
        total_score += count * weight
    return round(total_score, 2)


def get_recent_runs(limit=10):
    """7. Returns the latest runs for history view."""
    session = SESSION_LOCAL()
    try:
        runs = session.query(RunModel).order_by(RunModel.created_at.desc()).limit(limit).all()
        return [
            {
                "run_id": r.run_id,
                "mode": r.mode,
                "input_path": r.input_path,
                "output_path": r.output_path,
                "report_path": r.report_path,
                "created_at": r.created_at
            }
            for r in runs
        ]
    finally:
        session.close()
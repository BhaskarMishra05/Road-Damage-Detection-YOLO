import yaml
from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, ForeignKey, text, func, distinct
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
    """Creates the database tables if they do not exist."""
    BASE.metadata.create_all(bind=ENGINE)
    print("SUCCESS: Database tables initialized successfully via SQLAlchemy.")


def create_run(mode, input_path):
    """Creates a new run record and returns the new run_id."""
    with SESSION_LOCAL() as session:
        new_run = RunModel(mode=mode, input_path=input_path)
        session.add(new_run)
        session.commit()
        session.refresh(new_run)
        return new_run.run_id


def add_detections(run_id, detections):
    """Bulk insert detections using a list of dictionaries."""
    if not detections:
        return
    with SESSION_LOCAL() as session:
        session.bulk_save_objects([
            DetectionModel(
                run_id=run_id,
                track_id=d.get('track_id'),
                class_name=d['class_name'],
                confidence=d.get('confidence'),
                frame_no=d.get('frame_no')
            )
            for d in detections
        ])
        session.commit()


def update_run_outputs(run_id, output_path, report_path):
    """Fills in the result paths after processing finishes."""
    with SESSION_LOCAL() as session:
        run = session.get(RunModel, run_id)
        if run:
            run.output_path = output_path
            run.report_path = report_path
            session.commit()


def get_class_counts(run_id):
    """Returns defect counts per class directly via SQL aggregate queries to avoid memory overhead."""
    with SESSION_LOCAL() as session:
        run = session.get(RunModel, run_id)
        if not run:
            return {}
        
        if run.mode == 'image':
            results = session.query(
                DetectionModel.class_name,
                func.count(DetectionModel.det_id)
            ).filter(
                DetectionModel.run_id == run_id
            ).group_by(
                DetectionModel.class_name
            ).all()
        else:
            results = session.query(
                DetectionModel.class_name,
                func.count(distinct(DetectionModel.track_id))
            ).filter(
                DetectionModel.run_id == run_id,
                DetectionModel.track_id.isnot(None)
            ).group_by(
                DetectionModel.class_name
            ).all()
                
        return {class_name: count for class_name, count in results}


def get_severity_score(run_id):
    """Calculates severity score using hazard weights loaded from the configuration file."""
    import os
    default_weights = {
        'pothole': 5.0,
        'crack': 3.0,
        'debris': 2.5,
        'manhole': 2.0,
        'road barricade': 1.0,
        'speed breaker': 0.5
    }
    
    hazard_weights = default_weights
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(root_dir, "credientials", "config.ymal")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                if config and 'hazard_weights' in config:
                    hazard_weights = config['hazard_weights']
    except Exception:
        pass

    counts = get_class_counts(run_id)
    total_score = sum(count * hazard_weights.get(name.lower().strip(), 1.0) for name, count in counts.items())
    return round(total_score, 2)


def get_recent_runs(limit=10):
    """Returns the latest runs for history view."""
    with SESSION_LOCAL() as session:
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
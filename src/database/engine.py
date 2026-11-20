import os
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

def init_db():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(root_dir, "credientials", "config.ymal")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    db = config["database"]
    db_url = f"mysql+mysqlconnector://{db['user']}:{db['password']}@{db['host']}:{db['port']}/{db['name']}"
    
    engine = create_engine(db_url)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    return engine, SessionLocal, Base

ENGINE, SESSION_LOCAL, BASE = init_db()
import os
import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

def engin_creator():
    # Dynamically locate the project root directory
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Check potential folder spellings and filenames
    possible_folders = ['credientials', 'credentials']
    possible_filenames = ['config.ymal', 'config.yaml', 'database.yaml']
    config_path = None
    
    for folder in possible_folders:
        for filename in possible_filenames:
            path = os.path.join(root_dir, folder, filename)
            if os.path.exists(path):
                config_path = path
                break
        if config_path:
            break
            
    if not config_path:
        raise FileNotFoundError(
            f"\n[ERROR] Could not find configuration file inside 'credientials' or 'credentials'."
        )
        
    print(f"[INFO] Loading database credentials from: {config_path}")
    
    with open(config_path, 'r') as engine_obj:
        config = yaml.safe_load(engine_obj)
        
    config_database = config['database']

    db_url = f"mysql+mysqlconnector://{config_database['user']}:{config_database['password']}@{config_database['host']}:{config_database['port']}/{config_database['name']}"
    engine = create_engine(db_url)
    return engine

# Initialize engine, session maker, and base model for SQLAlchemy ORM
ENGINE = engin_creator()
SESSION_LOCAL = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)
BASE = declarative_base()
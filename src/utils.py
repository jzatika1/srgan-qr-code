import logging.config
import yaml
import os
import inspect

def setup_logging(default_path='config/logger.yaml', default_level=logging.INFO, env_key='LOG_CFG'):
    """Setup logging configuration"""
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

def get_logger(name):
    """Get a logger with a specified name"""
    logger = logging.getLogger(name)
    return logger

def get_function_logger():
    """Get a logger for the current function"""
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back
    func_name = caller_frame.f_code.co_name
    file_name = os.path.basename(caller_frame.f_code.co_filename).replace('.py', '')
    
    log_dir = os.path.join('logs', file_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'{func_name}.log')
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    
    logger = logging.getLogger(f'{file_name}.{func_name}')
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger

def save_model(model, model_name, model_dir='models/'):
    """Save the model to the specified directory"""
    try:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        model.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Failed to save model: {e}")

def load_model(model_name, model_dir='models/'):
    """Load a model from the specified directory"""
    from tensorflow.keras.models import load_model
    try:
        model_path = os.path.join(model_dir, f"{model_name}.h5")
        if os.path.exists(model_path):
            model = load_model(model_path)
            print(f"Model loaded from {model_path}")
            return model
        else:
            print(f"No model found at {model_path}")
            return None
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def save_config(config, config_path='config/config.yaml'):
    """Save the configuration to a YAML file"""
    try:
        with open(config_path, 'w') as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Failed to save configuration: {e}")

def load_config(config_path='config/config.yaml'):
    """Load configuration from a YAML file"""
    try:
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        print(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return None

def ensure_directories_exist(paths):
    """Ensure that the specified directories exist."""
    for path in paths:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory created at: {directory}")

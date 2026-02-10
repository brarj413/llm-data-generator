import logging
import pandas as pd
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def load_image_paths_from_metadata(metadata_path: Path, project_root_dir: Path, num_images: int = -1) -> List[Tuple[str, Path]]:
    try:
        df_metadata = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata from {metadata_path} with {len(df_metadata)} total entries.")
    except FileNotFoundError:
        logger.error(f"Metadata file not found: {metadata_path}")
        raise

    image_data_to_load = []
    for idx, row in df_metadata.iterrows():
        if num_images != -1 and len(image_data_to_load) >= num_images:
            break

        path_from_csv = row['image_path']
        img_absolute_path = project_root_dir / path_from_csv
        image_id = Path(path_from_csv).stem

        if img_absolute_path.exists():
            image_data_to_load.append((image_id, img_absolute_path))
        else:
            logger.warning(f"Image file not found using {img_absolute_path} path from csv metadata file.")

    if num_images != -1:
        logger.info(f"Selected {len(image_data_to_load)} images for inference.")
    else:
        logger.info(f"Selected all {len(image_data_to_load)} images for inference.")
    return image_data_to_load


def load_text_files_for_summarization(text_dir: Path, num_texts: int = -1) -> List[Tuple[str, str]]:
    text_data = []
    if not text_dir.exists():
        logger.warning(f"Text directory for summarization not found: {text_dir}")
        return text_data

    text_files = sorted(list(text_dir.glob("*.txt")))

    for txt_file_path in text_files:
        if num_texts != -1 and len(text_data) >= num_texts:
            break
        try:
            with open(txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            text_id = txt_file_path.stem
            text_data.append((text_id, content))
        except Exception as e:
            logger.warning(f"Could not read text file {txt_file_path}: {e}")

    if num_texts != -1:
        logger.info(f"Loaded {len(text_data)} texts for summarization (limit was {num_texts}).")
    else:
        logger.info(f"Loaded all {len(text_data)} texts for summarization from {text_dir}.")
    return text_data

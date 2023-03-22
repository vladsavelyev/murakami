"""
Reads all *.fb2 files from the ../data/murakami_fb2s directory and
concats them into one text file, and pushes them to Huggingface Hub
as `vldsavelyev/murakami` dataset repository.
"""

import os
from pathlib import Path
from lxml import etree
import datasets
from datasets import Dataset
from huggingface_hub import create_repo
import fire
import coloredlogs

coloredlogs.install(level="info")
datasets.logging.set_verbosity_info()


# Number of initial <p> element to take from each fb2, by number. This allows to skip
# intros and other junk in the beginning of an fb2. This is built semi-manually using
# the `helper_to_find_first_paragraphs` func.
START_PARAGRAPHS = {
    3: 5,
    6: 27,
    7: 3,
    9: 4,
    10: 3,
    12: 11,
    18: 5,
    20: 3,
    21: 5,
}


def helper_to_find_first_paragraphs(paragraphs, title, book_number, n=30):
    """
    Helps to eyeball first few paragraphs of a book to skip junk paragraphs
    in the beginning and manually construct the `tart_paragraphs` dict.
    """
    found_paragraphs = []
    skipping = True
    for i, p in enumerate(list(paragraphs)[:n]):
        if p.text is None:
            continue
        if book_number in START_PARAGRAPHS and i >= START_PARAGRAPHS[book_number]:
            skipping = False
        if skipping and p.text.lower() == title.lower():
            skipping = False
        if not skipping:
            found_paragraphs.append(f"   {i} {p.text}")

    if found_paragraphs:
        print("✅")
        print("\n".join(found_paragraphs))

    else:
        print("❌")
        for i, p in enumerate(list(paragraphs)[:30]):
            print(f"   {i} {p.text}")


def main(fb2_dir: Path, name: str = "murakami"):
    text_by_name = {}

    fb2s = list(Path(fb2_dir).glob("*.fb2"))
    if len(fb2s) > 0:
        print(f"Found {len(fb2s)} fb2 files in {fb2_dir}")
    else:
        raise ValueError(f"No fb2 files found in {fb2_dir}")

    for bi, path in enumerate(fb2s):
        print(bi, path)

        # Load the FB2 format file
        with path.open("rb") as file:
            fb2_data = file.read()

        # Print structure of the FB2 format file
        # print(etree.tostring(etree.fromstring(fb2_data), pretty_print=True))

        # Parse the FB2 format file using lxml
        root = etree.fromstring(fb2_data)

        # Get the title of the book
        title = root.xpath(
            "//fb:title-info/fb:book-title",
            namespaces={"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"},
        )[0].text
        print(title)

        # Get all book paragraphs
        paragraphs = root.xpath(
            "//fb:p",
            namespaces={"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"},
        )

        # UNCOMMENT THIS TO BUILD `START_PARAGRAPHS`
        # helper_to_find_first_paragraphs(paragraphs, title, bi)

        found_paragraphs = []
        skipping = True
        for pi, p in enumerate(paragraphs):
            if p.text is None:
                continue
            if bi in START_PARAGRAPHS and pi >= START_PARAGRAPHS[bi]:
                skipping = False
            if skipping and p.text.lower() == title.lower():
                skipping = False
            if not skipping:
                found_paragraphs.append(p)
        print(f"Found {len(found_paragraphs)} paragraphs")

        text_by_name[title] = ""
        for p in found_paragraphs:
            text_by_name[title] += p.text.replace(" ", " ") + "\n"
        text_by_name[title] += "\n"

    print("Novel by size:")
    for title, text in text_by_name.items():
        print(f"  {title}: {len(text):,} characters")

    smallest_title = min(text_by_name, key=lambda k: len(text_by_name[k]))
    print(
        f"Using smallest novel {smallest_title} "
        f"({len(text_by_name[smallest_title]):,} characters) as a test set"
    )
    train = Dataset.from_dict(
        {
            "text": [
                text_by_name[title] for title in text_by_name if title != smallest_title
            ]
        },
        split="train",
    )
    test = Dataset.from_dict(
        {"text": [text_by_name[smallest_title]]},
        split="test",
    )
    if token := os.getenv("HUB_TOKEN"):
        print(f"Pushing dataset to Huggingface Hub as dataset {name}...")
        create_repo(name, token=token, repo_type="dataset", exist_ok=True)
        train.push_to_hub(name, token=token)
        test.push_to_hub(name, token=token)
        print("Finished uploading dataset to Huggingface Hub")


if __name__ == "__main__":
    fire.Fire(main)

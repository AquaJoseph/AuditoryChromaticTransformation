import regex as re

def normalise_text(text: str) -> str:
    """
    Normalise text so dataframe names and filenames can be matched reliably.
    """
    text = str(text).strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[\\/:*?\"<>|]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def make_track_artist_stem(track: str, artist: str) -> str:
    track = normalise_text(track)
    artist = normalise_text(artist)
    return f"{track} - {artist}"
"""
Genre Bucket Mapping
====================
Maps Spotify genre strings (comma-separated) to broad musical buckets.

Broad buckets defined:
  classical, rock, metal, pop, hip_hop, punk,
  electronic, jazz, country, folk, soul_rnb, reggae, latin, blues, world

Usage as module:
    from genre_bucket_map import map_genres_to_buckets
    result = map_genres_to_buckets("rock, classic rock, baroque")
    # -> frozenset({'rock', 'classical'})

Usage as standalone script:
    python genre_bucket_map.py --input gigamidi_artist_prepared.csv
"""

import argparse
import re
from collections import Counter

# -----------------------------------------------------------------------------
# 0.  PRIORITY HIERARCHY
#     Used to break ties in majority voting. Rarer/more distinctive genres
#     are scored higher so they win against generic buckets like rock or pop.
# -----------------------------------------------------------------------------

PRIORITY = {
    "reggae": 15,
    "world": 14,
    "blues": 13,
    "latin": 12,
    "hip_hop": 11,
    "punk": 10,
    "electronic": 9,
    "metal": 8,
    "jazz": 7,
    "soul_rnb": 6,
    "country": 5,
    "folk": 4,
    "classical": 3,
    "rock": 2,
    "pop": 1
}

# -----------------------------------------------------------------------------
# 1.  EXCEPTIONS -- checked first (exact normalized genre name -> bucket)
#     Use this for genres that keyword rules would wrongly classify.
#     Value = bucket string, or None to discard that genre entirely.
# -----------------------------------------------------------------------------
EXCEPTIONS = {
    # "classic X" is NOT classical music
    "classic rock":            "rock",
    "classic pop":             "pop",
    "classic country":         "country",
    "classic country pop":     "country",
    "classic italian pop":     "pop",
    "classic schlager":        "pop",
    "classic soul":            "soul_rnb",
    "classic r&b":             "soul_rnb",
    "classic reggae":          "reggae",
    "classic hip hop":         "hip_hop",
    "classic blues":           "blues",
    "classic jazz":            "jazz",
    "classic funk":            "soul_rnb",
    "classic female blues":    "blues",
    "classic garagerock":      "rock",
    "classic colombian pop":   "pop",
    "classic czech pop":       "pop",
    "classic norwegian pop":   "pop",
    "classic polish pop":      "pop",
    "classic russian pop":     "pop",
    "classic turkish pop":     "pop",
    "classic uk pop":          "pop",

    # baroque/renaissance that is NOT classical music
    "baroque pop":             "pop",

    # neoclassical disambiguation
    "neoclassical metal":      "metal",
    "neoclassical dark wave":  "electronic",

    # folk disambiguation
    "folk metal":              "metal",
    "folk punk":               "punk",
    "folk rock":               "rock",
    "folk pop":                "folk",
    "celtic rock":             "rock",
    "celtic metal":            "metal",
    "celtic punk":             "punk",
    "new folk":                "folk",
    "traditional folk":        "folk",
    "contemporary folk":       "folk",
    "freak folk":              "folk",
    "anti-folk":               "folk",
    "acid folk":               "folk",
    "chamber folk":            "folk",
    "neofolk":                 "folk",

    # blues adjacent
    "blues rock":              "rock",
    "blues pop":               "pop",
    "country blues":           "blues",
    "soul blues":              "blues",
    "jazz blues":              "jazz",
    "rhythm and blues":        "soul_rnb",

    # jazz adjacent
    "bossa nova":              "jazz",
    "samba jazz":              "jazz",
    "jazz rock":               "jazz",
    "jazz fusion":             "jazz",
    "jazz pop":                "jazz",
    "jazz funk":               "jazz",
    "smooth jazz":             "jazz",
    "acid jazz":               "jazz",
    "nu jazz":                 "jazz",
    "free jazz":               "jazz",
    "chamber jazz":            "jazz",
    "avant-garde jazz":        "jazz",
    "vocal jazz":              "jazz",
    "contemporary jazz":       "jazz",
    "jazz rap":                "hip_hop",
    "jazz hip hop":            "hip_hop",
    "lounge":                  "jazz",

    # punk / hardcore disambiguation
    "metalcore":               "metal",
    "deathcore":               "metal",
    "grindcore":               "metal",
    "mathcore":                "metal",
    "electronicore":           "metal",
    "djent":                   "metal",
    "ska":                     "reggae",
    "ska punk":                "punk",
    "pop punk":                "punk",
    "skate punk":              "punk",
    "proto-punk":              "punk",
    "street punk":             "punk",
    "crust punk":              "punk",
    "cowpunk":                 "country",
    "psychobilly":             "rock",
    "horror punk":             "punk",
    "folk punk":               "punk",
    "anarcho-punk":            "punk",
    "christian punk":          "punk",
    "emo":                     "punk",
    "screamo":                 "punk",
    "post-hardcore":           "punk",
    "powerviolence":           "punk",
    "melodic hardcore":        "punk",
    "straight edge":           "punk",
    "nardcore":                "punk",

    # electronic adjacent
    "new wave":                "pop",
    "new wave pop":            "pop",
    "dark wave":               "electronic",
    "darkwave":                "electronic",
    "coldwave":                "electronic",
    "synthpop":                "electronic",
    "synth-pop":               "electronic",
    "electropop":              "electronic",
    "industrial rock":         "rock",
    "industrial metal":        "metal",
    "ebm":                     "electronic",
    "aggrotech":               "electronic",
    "futurepop":               "electronic",
    "lo-fi hip hop":           "hip_hop",
    "chillhop":                "hip_hop",
    "ambient pop":             "pop",
    "ambient folk":            "folk",
    "new age":                 "electronic",
    "new age music":           "electronic",
    "space ambient":           "electronic",
    "drone":                   "electronic",
    "noise":                   "electronic",
    "power electronics":       "electronic",
    "harsh noise":             "electronic",
    "dubstep":                 "electronic",
    "library music":           "electronic",
    "big beat":                "electronic",
    "trip hop":                "electronic",
    "downtempo":               "electronic",
    "electro":                 "electronic",
    "rave":                    "electronic",
    "solo wave":               "electronic",
    "hi-nrg":                  "electronic",

    # rock disambiguation
    "permanent wave":          "rock",
    "merseybeat":              "rock",
    "british invasion":        "rock",
    "new romantic":            "pop",
    "yacht rock":              "rock",
    "album rock":              "rock",
    "soft rock":               "rock",
    "hard rock":               "rock",
    "arena rock":              "rock",
    "mellow gold":             "pop",
    "adult standards":         "pop",
    "adult contemporary":      "pop",
    "easy listening":          "pop",
    "oldies":                  "pop",
    "doo-wop":                 "pop",
    "rockabilly":              "rock",
    "garage rock":             "rock",
    "surf rock":               "rock",
    "surf":                    "rock",
    "pub rock":                "rock",
    "protopunk":               "punk",
    "shred":                   "metal",
    "instrumental rock":       "rock",
    "rap rock":                "rock",
    "rap metal":               "metal",
    "funk metal":              "metal",
    "funk rock":               "rock",
    "post-grunge":             "rock",
    "grunge":                  "rock",
    "shoegaze":                "rock",
    "madchester":              "rock",
    "britpop":                 "rock",
    "baggy":                   "rock",
    "new wave of british heavy metal": "metal",
    "nwobhm":                  "metal",
    "alternative metal":       "metal",
    "supergroup":              "rock",
    "jam band":                "rock",
    "beatlesque":              "rock",
    "zolo":                    "rock",
    "beat italiano":           "rock",
    "australian psych":        "rock",
    "palm desert scene":       "rock",
    "spanish invasion":        "rock",

    # pop disambiguation
    "dance pop":               "pop",
    "dream pop":               "pop",
    "chamber pop":             "pop",
    "indie pop":               "pop",
    "art pop":                 "pop",
    "noise pop":               "pop",
    "bubblegum pop":           "pop",
    "teen pop":                "pop",
    "power pop":               "pop",
    "sunshine pop":            "pop",
    "sophisti-pop":            "pop",
    "jangle pop":              "pop",
    "twee pop":                "pop",
    "schlager":                "pop",
    "europop":                 "pop",
    "euro pop":                "pop",
    "eurodance":               "pop",
    "eurodisco":               "pop",
    "uk dance":                "electronic",
    "disco":                   "pop",
    "boogie":                  "pop",
    "nu-disco":                "electronic",
    "k-pop":                   "pop",
    "j-pop":                   "pop",
    "c-pop":                   "pop",
    "cantopop":                "pop",
    "mandopop":                "pop",
    "thai pop":                "pop",
    "city pop":                "pop",
    "shibuya-kei":             "pop",
    "kayokyoku":               "pop",
    "talent show":             "pop",
    "boy band":                "pop",
    "girl group":              "pop",
    "classic girl group":      "pop",
    "anime":                   "pop",
    "karneval":                "pop",
    "kolsche karneval":        "pop",
    "kermis":                  "pop",
    "ballroom":                "pop",
    "dutch cabaret":           "pop",
    "canzone d'autore":        "pop",
    "ye ye":                   "pop",
    "levenslied":              "pop",
    "tin pan alley":           "pop",

    # soul / R&B / gospel
    "neo soul":                "soul_rnb",
    "contemporary r&b":        "soul_rnb",
    "urban contemporary":      "soul_rnb",
    "quiet storm":             "soul_rnb",
    "new jack swing":          "soul_rnb",
    "gospel":                  "soul_rnb",
    "christian music":         "soul_rnb",
    "christian rock":          "rock",
    "christian metal":         "metal",
    "christian hip hop":       "hip_hop",
    "ccm":                     "soul_rnb",
    "worship":                 "soul_rnb",
    "funk":                    "soul_rnb",
    "p-funk":                  "soul_rnb",
    "go-go":                   "soul_rnb",
    "motown":                  "soul_rnb",
    "northern soul":           "soul_rnb",
    "blue-eyed soul":          "soul_rnb",

    # hip-hop
    "trap":                    "hip_hop",
    "uk drill":                "hip_hop",
    "grime":                   "hip_hop",
    "gangsta rap":             "hip_hop",
    "conscious hip hop":       "hip_hop",
    "alternative hip hop":     "hip_hop",
    "trap soul":               "hip_hop",
    "mumble rap":              "hip_hop",
    "cloud rap":               "hip_hop",
    "crunk":                   "hip_hop",
    "snap":                    "hip_hop",
    "horrorcore":              "hip_hop",
    "phonk":                   "hip_hop",
    "bounce":                  "hip_hop",
    "chicago drill":           "hip_hop",
    "plugg":                   "hip_hop",
    "dark trap":               "hip_hop",
    "lo-fi hip hop":           "hip_hop",
    "chillhop":                "hip_hop",

    # latin
    "latin pop":               "latin",
    "latin rock":              "latin",
    "latin alternative":       "latin",
    "latin jazz":              "jazz",
    "afro-cuban jazz":         "jazz",
    "salsa":                   "latin",
    "reggaeton":               "latin",
    "cumbia":                  "latin",
    "tango":                   "latin",
    "samba":                   "latin",
    "mpb":                     "latin",
    "forro":                   "latin",
    "pagode":                  "latin",
    "axe":                     "latin",
    "bolero":                  "latin",
    "ranchera":                "latin",
    "norteno":                 "latin",
    "corrido":                 "latin",
    "mariachi":                "latin",
    "merengue":                "latin",
    "bachata":                 "latin",
    "cha-cha-cha":             "latin",
    "mambo":                   "latin",
    "son cubano":              "latin",
    "nueva cancion":           "latin",
    "nueva trova":             "latin",
    "tropicalia":              "latin",
    "vallenato":               "latin",
    "sertanejo":               "latin",
    "baile funk":              "latin",
    "funk carioca":            "latin",
    "flamenco":                "world",

    # reggae
    "dancehall":               "reggae",
    "dub":                     "reggae",
    "roots reggae":            "reggae",
    "rocksteady":              "reggae",
    "lovers rock":             "reggae",
    "ragga":                   "reggae",
    "mento":                   "reggae",
    "calypso":                 "world",
    "soca":                    "world",

    # country
    "country road":            "country",
    "country rock":            "country",
    "country pop":             "country",
    "country rap":             "country",
    "bro-country":             "country",
    "outlaw country":          "country",
    "red dirt":                "country",
    "texas country":           "country",
    "nashville sound":         "country",
    "bluegrass":               "country",
    "alt-country":             "country",
    "contemporary country":    "country",

    # world / ethnic
    "afrobeat":                "world",
    "afrobeats":               "world",
    "highlife":                "world",
    "kwaito":                  "world",
    "amapiano":                "world",
    "soukous":                 "world",
    "bollywood":               "world",
    "hindustani":              "world",
    "carnatic":                "world",
    "qawwali":                 "world",
    "ghazal":                  "world",
    "klezmer":                 "world",
    "balkan":                  "world",
    "fado":                    "world",
    "gnawa":                   "world",
    "rai":                     "world",
    "zydeco":                  "world",
    "cajun":                   "world",
    "polka":                   "world",
    "j-rock":                  "rock",
    "j-metal":                 "metal",
    "visual kei":              "rock",
    "v-kei":                   "rock",
    "canzone napoletana":      "world",
    "canzone genovese":        "world",

    # specific exceptions
    "fingerstyle":             "folk",
    "violao":                  "folk",
    "cantautor":               "folk",
    "oktoberfest":             "folk",
    "classical guitar":        "classical",
    "electric bass":           "jazz",
    "smooth saxophone":        "jazz",

    # discard
    "test":                    None,
    "none":                    None,
    "unknown":                 None,
    "comic":                   None,
    "theme":                   None,
}

# -----------------------------------------------------------------------------
# 2.  KEYWORD RULES -- ordered, first match wins.
#     Applied ONLY if no exact match found in EXCEPTIONS.
#     (bucket, [list of substrings to search in normalized genre string])
# -----------------------------------------------------------------------------
KEYWORD_RULES = [
    # Metal first (before classical, so "neoclassical metal" -> metal)
    ("metal",      ["metal", "thrash", "deathcore", "metalcore", "mathcore",
                    "grindcore", "djent", "doom", "sludge", "shred", "nwobhm",
                    "heavy metal", "melodeath", "deathgrind", "goregrind",
                    "neue deutsche harte", "death 'n' roll", "black 'n' roll"]),

    # Classical -- "classical" != "classic"
    ("classical",  ["baroque", "renaissance", "classical", "early music",
                    "romantic", "medieval", "gregorian", "madrigal", "choral",
                    "musique concrete", "spectralism", "neoclassicism",
                    "impressionism", "chamber music", "orchestral", "oratorio",
                    "cantata", "liturgy", "school", "galante", "victorian",
                    "opera", "organ", "tenor", "pastoral"]),

    # Punk before rock
    ("punk",       ["punk", "hardcore", "screamo", "emo", "post-hardcore"]),

    # Hip-hop
    ("hip_hop",    ["hip hop", "hip-hop", "hiphop", "rap", "trap", "drill",
                    "grime", "boom bap", "crunk", "mumble rap", "cloud rap",
                    "phonk"]),

    # Blues
    ("blues",      ["blues"]),

    # Jazz
    ("jazz",       ["jazz", "swing", "bebop", "big band", "bossa nova", "fusion",
                    "post-bop", "ragtime", "hard bop", "dixieland"]),

    # Soul / R&B
    ("soul_rnb",   ["soul", "r&b", "rnb", "rhythm and blues", "motown",
                    "gospel", "funk", "urban contemporary", "quiet storm",
                    "minneapolis sound"]),

    # Country
    ("country",    ["country", "bluegrass", "americana", "honky tonk",
                    "outlaw", "cowboy", "western swing", "cosmic american",
                    "bakersfield sound"]),

    # Folk
    ("folk",       ["folk", "singer-songwriter", "acoustic", "chanson",
                    "troubadour", "skiffle", "lilith", "yodeling",
                    "musica lombarda", "musica sammarinese"]),

    # Latin
    ("latin",      ["latin", "salsa", "reggaeton", "cumbia", "tango", "samba",
                    "bolero", "ranchera", "corrido", "mariachi", "merengue",
                    "bachata", "forro", "pagode", "tropicalia", "nueva ",
                    "mpb", "vallenato", "trova", "cancion melodica",
                    "velha guarda", "jovem guarda", "tropical", "choro",
                    "sertanejo universitario"]),

    # Reggae
    ("reggae",     ["reggae", "ska", "dancehall", "dub", "rocksteady",
                    "ragga", "lovers rock"]),

    # Electronic
    ("electronic", ["electronic", "techno", "house", "trance", "edm",
                    "dubstep", "drum and bass", "dnb", "ambient", "synth",
                    "industrial", "idm", "breakbeat", "jungle", "chillwave",
                    "vaporwave", "lo-fi", "darkwave", "ebm", "noise",
                    "new age", "glitch", "drone", "experimental"]),

    # Rock (broad fallback)
    ("rock",       ["rock", "alternative", "indie", "shoegaze", "post-rock",
                    "math rock", "krautrock", "britpop", "madchester",
                    "grunge", "psychedelic", "prog", "glam", "new wave",
                    "permanent wave", "garage", "surf", "post-grunge",
                    "art rock", "space rock", "merseybeat", "punk rock",
                    "melancholia"]),

    # Pop (broad fallback)
    ("pop",        ["pop", "dance", "disco", "mellow gold", "adult standards",
                    "easy listening", "bubblegum", "schlager", "europop",
                    "chanson", "new romantic", "oldies", "doo-wop", "neo mellow",
                    "volksmusik", "neue deutsche welle", "dansktop"]),

    # World (catch-all for regional/ethnic)
    ("world",      ["world", "afrobeat", "afrobeats", "highlife", "kwaito",
                    "soukous", "bollywood", "hindustani", "carnatic",
                    "klezmer", "balkan", "fado", "gnawa", "rai", "flamenco",
                    "zydeco", "cajun", "polka", "celtic", "traditional",
                    "ethnic", "global", "international", "roots"]),
]


# -----------------------------------------------------------------------------
# 3.  Core mapping logic
# -----------------------------------------------------------------------------

def _normalize(genre: str) -> str:
    """Lowercase + strip + collapse whitespace."""
    return re.sub(r"\s+", " ", genre.strip().lower())


def map_single_genre(genre: str):
    """
    Map one individual genre string to a broad bucket.
    Returns a bucket name string, None (discard), or 'other' (unmapped).
    """
    norm = _normalize(genre)
    if not norm:
        return None

    # 1. Exact exception
    if norm in EXCEPTIONS:
        return EXCEPTIONS[norm]

    # 2. Keyword rules
    for bucket, keywords in KEYWORD_RULES:
        for kw in keywords:
            if kw in norm:
                return bucket

    return "other"


def map_genres_to_buckets(genre_string: str, allow_ties: bool = True) -> str:
    """
    Given a comma-separated genre string, return a SINGLE broad bucket string
    using Majority Voting + Distinctiveness Priority tie-breaker.
    Returns None if no genres could be mapped.
    If allow_ties=False, returns None if a tie occurs (only returns single/majority).
    """
    if not isinstance(genre_string, str) or not genre_string.strip():
        return None

    mapped_buckets = []
    for genre in genre_string.split(","):
        bucket = map_single_genre(genre.strip())
        if bucket and bucket != "other":
            mapped_buckets.append(bucket)
            
    if not mapped_buckets:
        return None
        
    counts = Counter(mapped_buckets)
    
    # Sort by (Count DESC, Priority DESC). This resolves ties by picking the bucket with higher Priority.
    sorted_buckets = sorted(
        counts.items(),
        key=lambda x: (x[1], PRIORITY.get(x[0], 0)),
        reverse=True
    )
    
    # Check for tie
    if not allow_ties and len(sorted_buckets) > 1:
        if sorted_buckets[0][1] == sorted_buckets[1][1]:
            return None  # Discard if it's a tie and ties are disabled
            
    return sorted_buckets[0][0]


# -----------------------------------------------------------------------------
# 4.  Standalone diagnostic script
# -----------------------------------------------------------------------------

def run_diagnostics(csv_path: str) -> None:
    import pandas as pd

    print(f"\nLoading '{csv_path}'...")
    df = pd.read_csv(csv_path, usecols=["matched_genres"], low_memory=False)

    bucket_track_counts = Counter()
    
    # Stats for how buckets were assigned
    # "single" -> All mapped subgenres belonged to the same bucket
    # "majority" -> Won by having strictly more votes
    # "tie" -> Won by Priority Hierarchy tiebreaker
    assign_stats = {b: {"single": 0, "majority": 0, "tie": 0} for b in PRIORITY.keys()}
    
    unmapped = 0
    total_valid = 0

    # Collect individual unmapped strings for the "other" report
    all_unmapped_tokens = []

    for cell in df["matched_genres"].dropna():
        genre_string = str(cell)
        if not genre_string.strip():
            continue
            
        mapped_buckets = []
        for genre in genre_string.split(","):
            b = map_single_genre(genre.strip())
            if b and b != "other":
                mapped_buckets.append(b)
            elif b == "other":
                all_unmapped_tokens.append(genre.strip().lower())
                
        if not mapped_buckets:
            unmapped += 1
            continue
            
        counts = Counter(mapped_buckets)
        sorted_buckets = sorted(
            counts.items(),
            key=lambda x: (x[1], PRIORITY.get(x[0], 0)),
            reverse=True
        )
        
        winning_bucket = sorted_buckets[0][0]
        bucket_track_counts[winning_bucket] += 1
        total_valid += 1
        
        if len(sorted_buckets) == 1:
            assign_stats[winning_bucket]["single"] += 1
        elif sorted_buckets[0][1] > sorted_buckets[1][1]:
            assign_stats[winning_bucket]["majority"] += 1
        else:
            assign_stats[winning_bucket]["tie"] += 1

    print(f"\nTotal songs evaluated      : {len(df):,}")
    print(f"Songs successfully mapped  : {total_valid:,}")
    print(f"Songs completely unmapped  : {unmapped:,}")

    print("\n-- 1-to-1 Bucket Distribution & Resolution Stats ---------------------")
    print(f"  {'Bucket':<15} {'Total Songs':>12} {'%':>6}  |  {'Single':>8} {'Majority':>8} {'Tie-Break':>9}")
    print("  " + "-" * 73)
    
    for bucket, count in bucket_track_counts.most_common():
        pct = (count / total_valid) * 100
        st = assign_stats[bucket]
        print(f"  {bucket:<15} {count:>12,} {pct:>5.1f}%  |  {st['single']:>8,} {st['majority']:>8,} {st['tie']:>9,}")

    if all_unmapped_tokens:
        print(f"\n-- Top 100 UNMAPPED individual subgenres ('other') -----------------")
        unmap_counts = Counter(all_unmapped_tokens)
        for g, count in unmap_counts.most_common(100):
            print(f"  {g:<30} {count:>6,}")
        if len(unmap_counts) > 100:
            print(f"  ... and {len(unmap_counts) - 100} more.")

    print("\n-- Sample mappings ---------------------------------------------------")
    samples = [
        "rock, classic rock, album rock, hard rock, british invasion",
        "baroque, early music, german baroque, classical",
        "metal, alternative metal, thrash metal, nu metal",
        "pop, dance pop, europop, new romantic",
        "hip hop, gangsta rap, trap, conscious hip hop",
        "classical, neoclassical metal",
        "bossa nova, jazz, mpb, samba jazz",
        "country, country road, contemporary country",
        "soul, r&b, neo soul, urban contemporary, gospel",
        "alternative metal, canadian pop punk, canadian punk, modern rock, pop punk, post-grunge, punk, rock",
    ]
    print()
    for s in samples:
        bucket = map_genres_to_buckets(s)
        print(f"  IN : '{s}'")
        print(f"  OUT: {bucket}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genre bucket mapping diagnostics")
    parser.add_argument("--input", default="gigamidi_matched_genres.csv",
                        help="Path to the matched genres CSV file")
    args = parser.parse_args()
    run_diagnostics(args.input)

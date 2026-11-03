"""The recogniser's hot-word list holds EVERY make and model of the database, written as they are written ("BMW", "CR-V", "Mazda3"), whatever the cars' status (ARCH_REVIEW section 97)."""
from pathlib import Path

import pytest

from dealer_core.voice.prompts import cased_names, hotword_names, proper_make, proper_model, stock_names
from dealer_core.voice.stt import HOTWORDS_SENT


@pytest.mark.parametrize("model,written", [
    ("cr-v", "CR-V"), ("zs", "ZS"), ("x-trail", "X-Trail"), ("santa fe", "Santa Fe"), ("rio", "Rio"), ("macan", "Macan"), ("land cruiser", "Land Cruiser"),
    ("rav4", "RAV4"), ("mazda3", "Mazda3"), ("cx-5", "CX-5"), ("k5", "K5"), ("g80", "G80"), ("qx60", "QX60"), ("m3", "M3"), ("x5", "X5"), ("a4", "A4"), ("911", "911"),
    ("320i", "320i"), ("530i", "530i"), ("es 350", "ES 350"), ("lx 600", "LX 600"), ("c 300", "C 300"), ("gle 450", "GLE 450"), ("g 63", "G 63"),
    ("model 3", "Model 3"), ("model y", "Model Y"), ("range rover sport", "Range Rover Sport"), ("grand cherokee", "Grand Cherokee"), ("accord", "Accord"),
])
def test_a_model_is_written_as_it_is_written(model, written):
    assert proper_model(model) == written


@pytest.mark.parametrize("make,written", [("bmw", "BMW"), ("gmc", "GMC"), ("mg", "MG"), ("land rover", "Land Rover"), ("toyota", "Toyota"), ("kia", "Kia"), ("audi", "Audi")])
def test_a_make_is_written_as_it_is_written(make, written):
    assert proper_make(make) == written


def test_the_whisper_hint_is_not_touched_by_the_new_forms():
    assert stock_names({"bmw": ["m3"], "honda": ["cr-v"]}) == ["Bmw", "M3", "Honda", "Cr-V"]            # section 82 pinned it: the hint is as it was


def test_every_make_and_model_is_in_the_list_each_once():
    vocab = {"honda": ["accord", "cr-v", "civic"], "porsche": ["macan", "cayenne", "911"], "bmw": ["320i", "m3"]}
    names = cased_names(vocab)
    assert names == ["Honda", "Accord", "CR-V", "Civic", "Porsche", "Macan", "Cayenne", "911", "BMW", "320i", "M3"]
    assert sorted(hotword_names(vocab, ["bmw"], 200)) == sorted(names)


def test_the_default_cap_covers_the_whole_stock_with_room_to_grow():
    assert HOTWORDS_SENT == 200


def test_every_car_in_the_database_whatever_its_status_is_in_the_list(db, inventory):
    """The inventory's hot-word vocabulary has no status filter: a car that was sold is still a car a caller may ask for. (vocabulary(), the NLU's, still holds the cars for sale only.)"""
    before = inventory.vocabulary()
    row = db.query("SELECT stock_id, make, model FROM cars WHERE status = 'available' LIMIT 1")[0]
    db.conn.execute("UPDATE cars SET status = 'sold' WHERE make = ? AND model = ?", (row["make"], row["model"]))
    pair = (row["make"].lower(), row["model"].lower())
    assert pair in {(mk, m) for mk, ms in inventory.hotword_vocabulary().items() for m in ms}                 # still heard
    assert pair not in {(mk, m) for mk, ms in inventory.vocabulary().items() for m in ms}                       # no longer for sale
    allpairs = {(r["make"].lower(), r["model"].lower()) for r in db.query("SELECT DISTINCT make, model FROM cars")}
    assert {(mk, m) for mk, ms in inventory.hotword_vocabulary().items() for m in ms} == allpairs
    assert len(cased_names(inventory.hotword_vocabulary())) <= HOTWORDS_SENT and before


def test_the_pipeline_gives_the_recogniser_the_every_car_vocabulary():
    import inspect

    from dealer_core.voice import pipeline

    assert "hotword_vocabulary" in inspect.getsource(pipeline.attach_stock_hotwords)

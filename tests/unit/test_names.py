"""Common first names in the name step (dealer_core/nlu/names.py; docs/ARCH_REVIEW.md section 28). Owner, 29 Oct: "it wasn't able to understand the name Ahmed";
callers will have Arab, white and Indian names."""
import sys
from pathlib import Path

import pytest

from dealer_core.nlu import names as N

sys.path.insert(0, str(Path(__file__).parent))


@pytest.mark.parametrize("heard, name", [("Achmed", "Ahmed"), ("Amed", "Ahmed"), ("Kalid", "Khalid"), ("Ayeesha", "Ayesha"), ("Deepack", "Deepak"),
                                         ("Jhon", "John"), ("Jonh", "John"), ("Hussien", "Hussein"), ("Annel", "Anil"), ("Ramish", "Ramesh"),
                                         ("Seed", "Saeed"), ("Ravie", "Ravi"), ("Muhamad", "Muhammad"), ("Priyah", "Priya")])
def test_a_near_miss_of_a_common_name_is_corrected(heard, name):
    assert N.correct(heard) == name


@pytest.mark.parametrize("name", ["Ahmed", "Ahmad", "Mohammed", "Muhammad", "Mohamed", "Fatima", "Fatma", "Fatema", "Aisha", "Ayesha", "Priya", "Lakshmi",
                                  "John", "Sarah", "Sara", "Omer", "Zayn", "Mohit"])
def test_a_listed_spelling_is_never_changed(name):
    assert N.correct(name) == name


@pytest.mark.parametrize("name", ["Raul", "Kenji", "Svetlana", "Tamara", "Ingrid", "Oluwaseun", "Ira", "Myra", "Kiara", "Adnan", "Freya", "Seren", "Ruqaya",
                                  "Hamza", "Zeeshan", "Rizwan", "Junaid", "Owen", "Callum", "Niamh", "Ishaan", "Aarav", "Anika", "Lujain", "Yazan",
                                  "Danish", "Shaima", "Samer", "Mazin"])
def test_real_names_are_left_alone(name):
    """Some are now on the list (added after they were found being "corrected": Danish became Dinesh, Shaima Saima, Tamara Tamer); every one must stay as said."""
    assert N.correct(name) == name


@pytest.mark.parametrize("heard", ["Kyrin", "Devia", "Sera", "Rame", "Edmmer", "Latch"])
def test_when_several_different_names_sound_alike_or_none_is_close_nothing_is_guessed(heard):
    assert N.correct(heard) == heard                                    # Kyrin: Karen, Kiran or Karan; Devia: Dave or Divya


def test_only_the_first_word_is_looked_at():
    assert N.correct("Achmed Khan") == "Ahmed Khan" and N.correct("Fatima Masood") == "Fatima Masood" and N.correct(None) is None


def test_the_name_step_writes_the_corrected_name_and_reads_it_back():
    import test_sweep_fixes as F

    e, s = F.at("name_q")
    intent, reply = F.say(e, s, "Achmed")
    assert s.draft_name == "Ahmed" and "for Ahmed" in reply and reply.endswith("Shall I book it?")

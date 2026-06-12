from data.ingestion import tournament_importance


def test_importance_tiers():
    assert tournament_importance("FIFA World Cup") == 5
    assert tournament_importance("FIFA World Cup qualification") == 3
    assert tournament_importance("UEFA Euro") == 4
    assert tournament_importance("UEFA Euro qualification") == 3
    assert tournament_importance("Copa América") == 4
    assert tournament_importance("UEFA Nations League") == 3
    assert tournament_importance("Friendly") == 1
    assert tournament_importance("Gold Cup") == 4
    assert tournament_importance("Some Random Trophy") == 2

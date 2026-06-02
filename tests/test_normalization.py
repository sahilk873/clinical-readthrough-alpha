"""Tests for normalization and fuzzy matching."""


from clinical_alpha.matching.normalizer import (
    create_matched_sponsor_map,
    fuzzy_match_score,
    match_sponsor_to_company,
    normalize_company_name,
    normalize_drug_name,
    resolve_best_match,
)


class TestNormalizeCompanyName:
    def test_removes_common_suffixes(self):
        assert normalize_company_name("Pfizer Inc") == normalize_company_name("Pfizer")
        assert normalize_company_name("Merck & Co Inc") == normalize_company_name("Merck")
        assert normalize_company_name("Johnson & Johnson") != ""

    def test_lowercases(self):
        assert normalize_company_name("PFIZER") == normalize_company_name("pfizer")

    def test_removes_punctuation(self):
        result = normalize_company_name("Bristol-Myers Squibb Co.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_string(self):
        assert normalize_company_name("") == ""
        assert normalize_company_name(None) == ""

    def test_handles_the(self):
        norm_with = normalize_company_name("The Company Inc")
        norm_without = normalize_company_name("Company")
        assert norm_with == norm_without


class TestNormalizeDrugName:
    def test_lowercases(self):
        assert normalize_drug_name("DRUG X") == "drug x"

    def test_removes_parentheses(self):
        assert normalize_drug_name("Drug (X)") == "drug x"

    def test_empty_string(self):
        assert normalize_drug_name("") == ""
        assert normalize_drug_name(None) == ""


class TestFuzzyMatch:
    def test_exact_match(self):
        assert fuzzy_match_score("Pfizer Inc", "Pfizer Inc") == 1.0

    def test_high_similarity(self):
        score = fuzzy_match_score("Pfizer Inc", "Pfizer")
        assert score > 0.7

    def test_low_similarity(self):
        score = fuzzy_match_score("Pfizer Inc", "Microsoft Corp")
        assert score < 0.5

    def test_empty_strings(self):
        assert fuzzy_match_score("", "") == 1.0
        assert fuzzy_match_score("a", "") == 0.0


class TestMatchSponsor:
    def test_returns_sorted_results(self, sample_company_df):
        results = match_sponsor_to_company("Pfizer Inc", sample_company_df)
        assert len(results) == len(sample_company_df)
        assert results[0]["ticker"] == "PFE"
        assert results[0]["score"] >= results[1]["score"]

    def test_high_confidence_threshold(self, sample_company_df):
        results = match_sponsor_to_company("Pfizer Inc", sample_company_df, threshold=0.9)
        high_conf = [r for r in results if r["high_confidence"]]
        assert len(high_conf) >= 1


class TestResolveBestMatch:
    def test_returns_best_match(self, sample_company_df):
        results = match_sponsor_to_company("Pfizer Inc", sample_company_df)
        best = resolve_best_match(results, min_score=0.7)
        assert best is not None
        assert best["ticker"] == "PFE"

    def test_no_match_below_threshold(self):
        results = [
            {"sponsor_raw": "Unknown Co", "ticker": "XXX", "score": 0.3, "high_confidence": False},
        ]
        best = resolve_best_match(results, min_score=0.7)
        assert best is None


class TestCreateSponsorMap:
    def test_creates_mapping(self, sample_company_df):
        sponsors = ["Pfizer Inc", "Merck & Co Inc", "Unknown Pharma Corp"]
        mapping = create_matched_sponsor_map(sponsors, sample_company_df, threshold=0.7)
        assert mapping["Pfizer Inc"] is not None
        assert mapping["Merck & Co Inc"] is not None
        assert (
            mapping["Unknown Pharma Corp"] is None
            or mapping["Unknown Pharma Corp"]["ticker"] != "PFE"
        )

    def test_unknown_sponsor_is_none(self, sample_company_df):
        mapping = create_matched_sponsor_map(
            ["Totally Unknown Company"], sample_company_df, threshold=0.9
        )
        assert mapping["Totally Unknown Company"] is None

from sentimental.pipeline import clean_text


def test_converts_input_to_lowercase():
    """ This function will check if clean_text converts input to lower case """
    text = "Simple text with Title case, ALL-CAPS and normal lowercase"
    output = clean_text(text)
    assert output.islower()


def test_removes_unicode_object():
    """ This function will check if clean_text removes Unicode objects """
    text = "No Unicode #x2F063B; object here"
    output = clean_text(text)
    assert output == "no unicode object here"


def test_removes_web_urls_with_www():
    """ This function will check if clean_text removes Web URLs starting with www """
    text = "No Web URL www.gmail.com here"
    output = clean_text(text)
    assert output == "no web url here"


def test_removes_web_urls_with_http():
    """ This function will check if clean_text removes Web URLs starting with http """
    text = "No Web URL http://www.gmail.com here"
    output = clean_text(text)
    assert output == "no web url here"


def test_removes_web_urls_with_https():
    """ This function will check if clean_text removes Web URLs starting with https """
    text = "No Web URL https://www.gmail.com here"
    output = clean_text(text)
    assert output == "no web url here"


def test_removes_html_tags():
    """ This function will check if clean_text removes HTML tags """
    text = "No <a href='hyperlink' target='_blank'> HTML tag here"
    output = clean_text(text)
    assert output == "no html tag here"


def test_removes_html_objects_with_ampersands():
    """ This function will check if clean_text removes HTML objects """
    text = "No &nbsp; HTML object &amp; here"
    output = clean_text(text)
    assert output == "no html object here"


def test_removes_html_objects_without_ampersands():
    """ This function will check if clean_text removes HTML objects """
    text = "No nbsp; HTML object amp; here"
    output = clean_text(text)
    assert output == "no html object here"


def test_removes_user_handles():
    """ This function will check if clean_text removes user handles """
    text = "No User @some_user handle @another here"
    output = clean_text(text)
    assert output == "no user handle here"


def test_removes_hashtag_prefix():
    """ This function will check if clean_text removes hashtags (#) prefix """
    text = "Hashtag #Save_Earth do not have prefix and #MeToo"
    output = clean_text(text)
    assert output == "hashtag save_earth do not have prefix and metoo"


def test_restores_expanded_text():
    """ This function will check if clean_text restores expanded text (e.g. f l o r i d a) """
    text = "Expanded  t w i t t e r  Texts  n e v e r  belong here"
    output = clean_text(text)
    assert output == "expanded twitter texts never belong here"


def test_removes_extra_whitespace():
    """ This function will check if clean_text removes extra whitespace """
    text = """Extra spaces       and
    new
    
    
    lines are     strongly discouraged"""
    output = clean_text(text)
    assert output == "extra spaces and new lines are strongly discouraged"


def test_removes_trailing_punctuations():
    """ This function will check if clean_text removes trailing punctuations """
    text = "Punctuations anyone?! Sure: and bring lots of them, too!!!"
    output = clean_text(text)
    assert output == "punctuations anyone sure and bring lots of them too"


def test_removes_words_with_non_ascii_characters():
    """ This function will check if clean_text removes words with non-ascii characters """
    text = "Only Iâm ASCII characters are thisð allowed"
    output = clean_text(text)
    assert output == "only ascii characters are allowed"

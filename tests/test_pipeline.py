from sentimental.pipeline import clean_text


def test_converts_input_to_lowercase():
    """ This function will check if clean_text converts input to lower case """
    text = "Simple text with Title case, ALL-CAPS and normal lowercase"
    output = clean_text(text)
    assert output.islower()


def test_removes_unicode_literal():
    """ This function will check if clean_text removes Unicode literals """
    text = "No Unicode literal \u003F here"
    output = clean_text(text)
    assert output == "no unicode literal here"


def test_removes_unicode_object():
    """ This function will check if clean_text removes Unicode objects """
    text = "No Unicode #x2F063B object here"
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

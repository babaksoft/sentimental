# Preprocessing pipeline (summary)

## Overview
Pipeline v1 applies general-purpose text cleaning focused on syntactic noise reduction
prior to TF-IDF feature extraction.

## Applied transforms (in order)
- Convert to lowercase
- Remove Unicode strings
  - Example : #x003F;
- Remove Web URLs
  - Examples : www. | http:// | https://
- Remove HTML tags
- Remove HTML objects
  - Examples : &amp; | nbsp;
- Remove user handles
  - Example : @UserName
- Remove hashtag prefix from hashtags
- Replace expanded text with normal form
  - Example : t w i t t e r -> twitter
- Remove trailing punctuation character(s)
- Replace extra whitespace and new-line with a single space
- Remove tokens with non-ASCII characters (including Unicode emojis like 🥲)
- Transform with TfidfVectorizer

## Known limitations
Current pipeline version (v1) :
- lacks semantic normalization (e.g. no lemmatization)
- removes emojis that may carry sentiment cues
- removes expressive punctuation intensity (e.g. Amazing!!!!!)
- no domain-specific slang normalization (e.g. idk -> i_don't_know)
- does not handle negations or multi-word phrases

## Potential improvements (Pipeline v2)
- Lemmatization + n-gram phrase detection
- selective emoji-to-text translation
- negation binding (e.g. not_happy)
- mental health abbreviation dictionary
- configurable preprocessing profiles (e.g. strict vs light)

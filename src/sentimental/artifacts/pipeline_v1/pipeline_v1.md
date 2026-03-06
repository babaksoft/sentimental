# Preprocessing pipeline (summary)

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
- Remove tokens with non-ASCII characters
- Transform with TfidfVectorizer

## Spacial cases
- Remove Unicode emojis
  - Example : 🥲
- Keep character emoticons
  - Examples : (: | ;-) | :D | :/

/**
* @file text_analysis.h
 * @brief Text analysis for BM25 (Bm25Params): Unicode-aware tokens, English stop words,
 * and the Snowball English (Porter2) stemmer.
 */
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace vecdb {

    /// The Snowball English stemmer ("Porter2", as in snowball's algorithms/english.sbl):
    /// "running" -> "run", "generously" -> "generous". Expects a lowercase word; letters
    /// other than a to z are kept as they are (consonants to the algorithm).
    std::string stem_english(std::string_view word);

    /// Tokens of UTF-8 text: runs of letters and digits, lowercased (ASCII, Latin-1, Latin
    /// Extended-A, Greek, Cyrillic). Spaces, punctuation (ASCII and Unicode), symbols,
    /// emoji, control characters and invalid UTF-8 separate tokens. An apostrophe (' or
    /// U+2019) between letters stays in the token, as "'": "don't", "john's".
    std::vector<std::string> tokenize_unicode(std::string_view text);

    /// Whether a lowercase token is in Snowball's English stop-word list (174 words).
    bool is_english_stop_word(std::string_view token);

}  // namespace vecdb
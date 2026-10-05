/**
 * @file metadata.h
 * @brief Metadata attached to vectors: field types, values, schemas, and the
 *        column storage used by filtered search.
 */
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "payload_index.h"

namespace vecdb {

/// The type of a metadata field.
enum class FieldType : std::uint8_t { Int, Float, Bool, Keyword, Tags };

/// Readable name of a field type, for error messages.
inline const char* field_type_name(FieldType t) {
    switch (t) {
        case FieldType::Int: return "integer";
        case FieldType::Float: return "float";
        case FieldType::Bool: return "boolean";
        case FieldType::Keyword: return "keyword";
        default: return "tag set";
    }
}

/**
 * @brief One metadata value: an integer, float, boolean, keyword (string) or tag set.
 *
 * Built with Value::from(x), which picks the type from the C++ type of `x`:
 * bool -> Bool, any other integer -> Int, floating point -> Float,
 * std::vector<std::string> -> Tags, anything convertible to a string -> Keyword.
 */
struct Value {
    FieldType type = FieldType::Int;
    std::int64_t i = 0;
    double f = 0.0;
    bool b = false;
    std::string s;
    std::vector<std::string> tags;

    static Value of_int(std::int64_t v) { Value x; x.type = FieldType::Int; x.i = v; return x; }
    static Value of_float(double v) { Value x; x.type = FieldType::Float; x.f = v; return x; }
    static Value of_bool(bool v) { Value x; x.type = FieldType::Bool; x.b = v; return x; }
    static Value of_keyword(std::string v) { Value x; x.type = FieldType::Keyword; x.s = std::move(v); return x; }
    static Value of_tags(std::vector<std::string> v) { Value x; x.type = FieldType::Tags; x.tags = std::move(v); return x; }

    /// Builds a Value from a C++ value, choosing the type as described above.
    template <class T>
    static Value from(T&& v) {
        using D = std::decay_t<T>;
        if constexpr (std::is_same_v<D, Value>) return std::forward<T>(v);
        else if constexpr (std::is_same_v<D, bool>) return of_bool(v);
        else if constexpr (std::is_integral_v<D>) return of_int(static_cast<std::int64_t>(v));
        else if constexpr (std::is_floating_point_v<D>) return of_float(static_cast<double>(v));
        else if constexpr (std::is_same_v<D, std::vector<std::string>>) return of_tags(std::forward<T>(v));
        else return of_keyword(std::string(std::string_view(v)));
    }

    /// Equal type and value. Tag sets compare as sets (order and repeats ignored).
    friend bool operator==(const Value& lhs, const Value& rhs) {
        if (lhs.type != rhs.type) return false;
        switch (lhs.type) {
            case FieldType::Int: return lhs.i == rhs.i;
            case FieldType::Float: return lhs.f == rhs.f;
            case FieldType::Bool: return lhs.b == rhs.b;
            case FieldType::Keyword: return lhs.s == rhs.s;
            default: {
                auto x = lhs.tags, y = rhs.tags;
                std::sort(x.begin(), x.end());
                x.erase(std::unique(x.begin(), x.end()), x.end());
                std::sort(y.begin(), y.end());
                y.erase(std::unique(y.begin(), y.end()), y.end());
                return x == y;
            }
        }
    }
};

/**
 * @brief The metadata of one vector: field name -> value.
 *
 * Built fluently: Metadata().set("category", "news").set("year", 2024)
 *                           .set_tags("tags", {"ai", "chips"}).
 * unset(name) is used with set_metadata() to clear a field.
 */
class Metadata {
public:
    /// Sets a field; the type comes from the C++ type of `value` (see Value::from).
    template <class T>
    Metadata& set(std::string name, T&& value) {
        fields_[std::move(name)] = Value::from(std::forward<T>(value));
        return *this;
    }
    /// Sets a tag-set field.
    Metadata& set_tags(std::string name, std::vector<std::string> tags) {
        fields_[std::move(name)] = Value::of_tags(std::move(tags));
        return *this;
    }
    /// Marks a field to be cleared (meaningful for set_metadata()).
    Metadata& unset(std::string name) {
        fields_[std::move(name)] = std::nullopt;
        return *this;
    }

    /// The value of a field, or nothing if it is absent or marked unset.
    std::optional<Value> get(std::string_view name) const {
        auto it = fields_.find(std::string(name));
        return it == fields_.end() ? std::nullopt : it->second;
    }
    bool empty() const { return fields_.empty(); }
    std::size_t size() const { return fields_.size(); }
    const std::map<std::string, std::optional<Value>>& fields() const { return fields_; }

    friend bool operator==(const Metadata& a, const Metadata& b) { return a.fields_ == b.fields_; }

private:
    std::map<std::string, std::optional<Value>> fields_;  // nullopt = unset
};

/// A declared field: its name and type.
struct FieldSpec {
    std::string name;
    FieldType type;
    friend bool operator==(const FieldSpec&, const FieldSpec&) = default;
};

/// Strict: only declared fields are allowed. Dynamic: new fields are created on first use.
enum class SchemaMode { Strict, Dynamic };

/**
 * @brief Which metadata fields an index accepts.
 *
 *  - Schema::strict({...}): only these fields; unknown fields are rejected at
 *    insert, and a filter on an unknown field is an error.
 *  - Schema::dynamic({...}): these fields plus any new field, created on first
 *    use with the type of its first value. A filter on an unknown field matches
 *    nothing (the field may simply not exist yet).
 *
 * In both modes a field keeps one type: a value of another type is rejected.
 * Throws std::invalid_argument for an empty or duplicate field name.
 */
class Schema {
public:
    static Schema strict(std::vector<FieldSpec> fields) { return Schema(SchemaMode::Strict, std::move(fields)); }
    static Schema dynamic(std::vector<FieldSpec> fields = {}) { return Schema(SchemaMode::Dynamic, std::move(fields)); }

    SchemaMode mode() const { return mode_; }
    const std::vector<FieldSpec>& fields() const { return fields_; }

private:
    Schema(SchemaMode mode, std::vector<FieldSpec> fields) : mode_(mode), fields_(std::move(fields)) {
        for (std::size_t i = 0; i < fields_.size(); ++i) {
            if (fields_[i].name.empty()) throw std::invalid_argument("field name must not be empty");
            for (std::size_t j = 0; j < i; ++j)
                if (fields_[j].name == fields_[i].name)
                    throw std::invalid_argument("duplicate field name: " + fields_[i].name);
        }
    }

    SchemaMode mode_;
    std::vector<FieldSpec> fields_;
};

/**
 * @brief Column storage for metadata, one row per slot (NodeId).
 *
 * Layout:
 *  - One column per field, holding only that field's type: int64, double, a
 *    byte per boolean, a dictionary ID per keyword, a list of dictionary IDs per
 *    tag set; plus a "has a value" byte per row.
 *  - Strings are stored once in a dictionary and referenced by 32-bit IDs, so
 *    keyword and tag comparisons are integer comparisons.
 *  - Columns grow only up to the highest slot written, so unused fields and
 *    unwritten rows cost nothing.
 *
 * Slot operations mirror the indexes' slot operations: write (insert, slot
 * reuse), update (set_metadata), clear (remove), move_row (swap-with-last).
 *
 * Exception safety: validate() changes nothing. write() and update() are
 * all-or-nothing: everything that can throw (creating fields, adding strings,
 * growing columns) happens before any visible change. clear(), move_row() and
 * undo_write() never throw.
 *
 * Indexed fields (keyword, boolean, tag set) keep exact per-value counts for
 * the query planner; see ValueCounts.
 */
class MetadataStore {
public:
    /// Remembers what write() created, so undo_write() can remove it.
    struct WriteUndo {
        std::size_t fields;
        std::size_t strings;
    };

    explicit MetadataStore(Schema schema = Schema::dynamic()) : mode_(schema.mode()) {
        for (const FieldSpec& spec : schema.fields()) create_field(spec.name, spec.type);
    }

    SchemaMode mode() const { return mode_; }
    std::size_t field_count() const { return specs_.size(); }
    const std::string& field_name(std::size_t f) const { return specs_[f].name; }
    FieldType field_type(std::size_t f) const { return specs_[f].type; }

    /// Column number of a field, or nothing if it does not exist.
    std::optional<std::size_t> find_field(std::string_view name) const {
        auto it = by_name_.find(std::string(name));
        if (it == by_name_.end()) return std::nullopt;
        return it->second;
    }

    /// The schema as it is now: declared fields plus fields created dynamically.
    Schema current_schema() const {
        return mode_ == SchemaMode::Strict ? Schema::strict(specs_) : Schema::dynamic(specs_);
    }

    /// Number of distinct strings stored (keywords and tags).
    std::size_t dictionary_size() const { return strings_.size(); }
    /// Dictionary ID of a string, or nothing if no vector uses it.
    std::optional<std::uint32_t> find_string(std::string_view s) const {
        auto it = string_ids_.find(std::string(s));
        if (it == string_ids_.end()) return std::nullopt;
        return it->second;
    }
    const std::string& string_at(std::uint32_t id) const { return strings_[id]; }

    /// Checks `md` against the schema without changing anything. Throws
    /// std::invalid_argument for an unknown field (strict mode), a type that
    /// differs from the field's type, or a float that is NaN or infinite.
    /// An integer value is accepted for a float field.
    void validate(const Metadata& md) const {
        for (const auto& [name, value] : md.fields()) {
            if (name.empty()) throw std::invalid_argument("field name must not be empty");
            if (value && value->type == FieldType::Float && !std::isfinite(value->f))
                throw std::invalid_argument("metadata field '" + name + "' is NaN or infinite");
            const auto f = find_field(name);
            if (!f) {
                if (mode_ == SchemaMode::Strict) throw std::invalid_argument("unknown metadata field: " + name);
                continue;
            }
            if (value && !compatible(specs_[*f].type, value->type))
                throw std::invalid_argument("metadata field '" + name + "' is " + field_type_name(specs_[*f].type) +
                                            ", got " + field_type_name(value->type));
        }
    }

    /// Replaces the whole row of `slot`: given fields are set, all others cleared.
    /// All-or-nothing. Returns what to pass to undo_write() to roll it back.
    WriteUndo write(NodeId slot, const Metadata& md) { return apply(slot, md, true); }

    /// Changes only the given fields of `slot` (unset() clears one). All-or-nothing.
    void update(NodeId slot, const Metadata& md) { apply(slot, md, false); }

    /// Rolls back a write() that was the most recent change: clears the row and
    /// removes fields and strings that write created. Never throws.
    void undo_write(NodeId slot, const WriteUndo& undo) noexcept {
        clear(slot);
        truncate(undo);
    }

    /// Clears every field of `slot`. Never throws.
    void clear(NodeId slot) noexcept {
        for (Column& c : cols_) c.clear_row(slot);
    }

    /// Moves row `from` into row `to` (swap-with-last removal; `to` < `from`)
    /// and clears `from`. Never throws.
    void move_row(NodeId from, NodeId to) noexcept {
        for (Column& c : cols_) c.move_row(from, to);
    }

    /// The metadata of `slot` (only fields that have a value).
    Metadata read(NodeId slot) const {
        Metadata md;
        for (std::size_t f = 0; f < cols_.size(); ++f) {
            if (!has(f, slot)) continue;
            const Column& c = cols_[f];
            switch (c.type) {
                case FieldType::Int: md.set(specs_[f].name, c.ints[slot]); break;
                case FieldType::Float: md.set(specs_[f].name, c.floats[slot]); break;
                case FieldType::Bool: md.set(specs_[f].name, c.bools[slot] != 0); break;
                case FieldType::Keyword: md.set(specs_[f].name, strings_[c.keywords[slot]]); break;
                case FieldType::Tags: {
                    std::vector<std::string> tags;
                    for (std::uint32_t id : c.tags[slot]) tags.push_back(strings_[id]);
                    md.set_tags(specs_[f].name, std::move(tags));
                    break;
                }
            }
        }
        return md;
    }

    // ---- Fast accessors used by compiled filters -----------------------------

    bool has(std::size_t f, NodeId slot) const {
        const Column& c = cols_[f];
        return slot < c.present.size() && c.present[slot] != 0;
    }
    std::int64_t int_at(std::size_t f, NodeId slot) const { return cols_[f].ints[slot]; }
    double float_at(std::size_t f, NodeId slot) const { return cols_[f].floats[slot]; }
    bool bool_at(std::size_t f, NodeId slot) const { return cols_[f].bools[slot] != 0; }
    std::uint32_t keyword_at(std::size_t f, NodeId slot) const { return cols_[f].keywords[slot]; }
    const std::vector<std::uint32_t>& tags_at(std::size_t f, NodeId slot) const { return cols_[f].tags[slot]; }

    // ---- Payload index ------------------------------------------------------

    /// Starts keeping exact per-value counts for a keyword, boolean or tag-set
    /// field, counting existing rows. Throws std::invalid_argument for an unknown
    /// field or a numeric field. Doing it twice is harmless. All-or-nothing.
    void index_field(std::string_view name) {
        const auto f = find_field(name);
        if (!f) throw std::invalid_argument("cannot index unknown field: " + std::string(name));
        Column& c = cols_[*f];
        if (c.type == FieldType::Int || c.type == FieldType::Float)
            throw std::invalid_argument("only keyword, boolean and tag-set fields can be indexed");
        if (c.indexed) return;
        c.counts.reset(c.type == FieldType::Bool ? 2 : strings_.size());  // may throw; nothing changed yet
        c.indexed = true;
        for (NodeId s = 0; s < c.present.size(); ++s) c.count_row(s, true);
    }
    bool is_indexed(std::size_t f) const { return cols_[f].indexed; }
    /// Names of indexed fields.
    std::vector<std::string> indexed_fields() const {
        std::vector<std::string> out;
        for (std::size_t f = 0; f < cols_.size(); ++f)
            if (cols_[f].indexed) out.push_back(specs_[f].name);
        return out;
    }
    /// Live vectors holding value `id` in indexed field `f`, or nothing if not indexed.
    std::optional<std::size_t> value_count(std::size_t f, std::uint32_t id) const {
        if (!cols_[f].indexed) return std::nullopt;
        return cols_[f].counts.count(id);
    }
    /// Live vectors that have any value in field `f` (always tracked).
    std::size_t present_count(std::size_t f) const { return cols_[f].present_count; }

private:
    /// One field's storage.
    struct Column {
        FieldType type;
        bool indexed = false;
        std::size_t present_count = 0;
        std::vector<std::uint8_t> present;
        std::vector<std::int64_t> ints;
        std::vector<double> floats;
        std::vector<std::uint8_t> bools;
        std::vector<std::uint32_t> keywords;
        std::vector<std::vector<std::uint32_t>> tags;
        ValueCounts counts;

        explicit Column(FieldType t) : type(t) {}
        std::size_t rows() const { return present.size(); }

        /// Grows to at least `n` rows. May throw; extra rows have no value.
        void ensure(std::size_t n) {
            if (present.size() >= n) return;
            switch (type) {
                case FieldType::Int: ints.resize(n); break;
                case FieldType::Float: floats.resize(n); break;
                case FieldType::Bool: bools.resize(n); break;
                case FieldType::Keyword: keywords.resize(n); break;
                case FieldType::Tags: tags.resize(n); break;
            }
            present.resize(n, 0);  // last, so rows() only grows once every array has
        }

        /// Adds or removes row `s`'s value(s) from the counts of an indexed column.
        void count_row(NodeId s, bool add) noexcept {
            if (!indexed || s >= rows() || !present[s]) return;
            auto apply_one = [&](std::uint32_t id) { add ? counts.add(id) : counts.remove(id); };
            switch (type) {
                case FieldType::Bool: apply_one(bools[s]); break;
                case FieldType::Keyword: apply_one(keywords[s]); break;
                case FieldType::Tags:
                    for (std::uint32_t id : tags[s]) apply_one(id);
                    break;
                default: break;
            }
        }

        void clear_row(NodeId s) noexcept {
            if (s >= rows() || !present[s]) return;
            count_row(s, false);
            present[s] = 0;
            --present_count;
            if (type == FieldType::Tags) tags[s].clear();
        }

        void move_row(NodeId from, NodeId to) noexcept {
            clear_row(to);
            if (from >= rows() || !present[from]) return;
            if (to >= rows()) {  // cannot happen when to < from; drop the value safely
                clear_row(from);
                return;
            }
            // The value moves between two rows: counts and present_count stay the same.
            present[to] = 1;
            present[from] = 0;
            switch (type) {
                case FieldType::Int: ints[to] = ints[from]; break;
                case FieldType::Float: floats[to] = floats[from]; break;
                case FieldType::Bool: bools[to] = bools[from]; break;
                case FieldType::Keyword: keywords[to] = keywords[from]; break;
                case FieldType::Tags:
                    tags[to].swap(tags[from]);
                    tags[from].clear();
                    break;
            }
        }
    };

    /// A value resolved for one column, ready to commit without allocating.
    struct Pending {
        std::size_t field;
        bool clear = false;
        std::int64_t i = 0;
        double f = 0.0;
        bool b = false;
        std::uint32_t id = 0;
        std::vector<std::uint32_t> tags;
    };

    static bool compatible(FieldType field, FieldType value) {
        return field == value || (field == FieldType::Float && value == FieldType::Int);
    }

    /// Appends a field. May throw; on failure nothing changed.
    void create_field(const std::string& name, FieldType type) {
        specs_.reserve(specs_.size() + 1);
        cols_.reserve(cols_.size() + 1);
        by_name_.emplace(name, specs_.size());  // may throw; nothing changed yet
        specs_.push_back({name, type});         // cannot throw: reserved
        cols_.emplace_back(type);
    }

    /// Dictionary ID of `s`, adding it if new. May throw; on failure nothing changed.
    std::uint32_t intern(const std::string& s) {
        if (auto it = string_ids_.find(s); it != string_ids_.end()) return it->second;
        const auto id = static_cast<std::uint32_t>(strings_.size());
        strings_.push_back(s);
        try {
            string_ids_.emplace(s, id);
        } catch (...) {
            strings_.pop_back();
            throw;
        }
        return id;
    }

    /// Removes fields and strings added after `undo` was taken. Never throws.
    void truncate(const WriteUndo& undo) noexcept {
        while (specs_.size() > undo.fields) {
            by_name_.erase(specs_.back().name);
            specs_.pop_back();
            cols_.pop_back();
        }
        while (strings_.size() > undo.strings) {
            string_ids_.erase(strings_.back());
            strings_.pop_back();
        }
    }

    WriteUndo apply(NodeId slot, const Metadata& md, bool replace) {
        validate(md);
        const WriteUndo undo{specs_.size(), strings_.size()};
        std::vector<Pending> pending;

        // Phase 1: everything that can throw. Nothing visible changes yet.
        try {
            pending.reserve(md.size());
            for (const auto& [name, value] : md.fields()) {
                auto f = find_field(name);
                if (!f) {
                    if (!value) continue;  // clearing a field that does not exist yet
                    create_field(name, value->type);
                    f = specs_.size() - 1;
                }
                Pending p;
                p.field = *f;
                if (!value) {
                    p.clear = true;
                } else {
                    switch (specs_[*f].type) {
                        case FieldType::Int: p.i = value->i; break;
                        case FieldType::Float:
                            p.f = value->type == FieldType::Int ? static_cast<double>(value->i) : value->f;
                            break;
                        case FieldType::Bool: p.b = value->b; break;
                        case FieldType::Keyword: p.id = intern(value->s); break;
                        case FieldType::Tags:
                            for (const std::string& t : value->tags) {
                                const std::uint32_t id = intern(t);
                                if (std::find(p.tags.begin(), p.tags.end(), id) == p.tags.end()) p.tags.push_back(id);
                            }
                            break;
                    }
                }
                pending.push_back(std::move(p));
            }
            for (const Pending& p : pending) {
                Column& c = cols_[p.field];
                if (p.clear) continue;
                c.ensure(std::size_t(slot) + 1);
                if (c.indexed) c.counts.ensure(c.type == FieldType::Bool ? 2 : strings_.size());
            }
        } catch (...) {
            truncate(undo);
            throw;
        }

        // Phase 2: commit. Nothing below allocates or throws.
        if (replace) clear(slot);
        for (Pending& p : pending) {
            Column& c = cols_[p.field];
            c.clear_row(slot);
            if (p.clear) continue;
            switch (c.type) {
                case FieldType::Int: c.ints[slot] = p.i; break;
                case FieldType::Float: c.floats[slot] = p.f; break;
                case FieldType::Bool: c.bools[slot] = p.b ? 1 : 0; break;
                case FieldType::Keyword: c.keywords[slot] = p.id; break;
                case FieldType::Tags: c.tags[slot].swap(p.tags); break;
            }
            c.present[slot] = 1;
            ++c.present_count;
            c.count_row(slot, true);
        }
        return undo;
    }

    SchemaMode mode_;
    std::vector<FieldSpec> specs_;
    std::vector<Column> cols_;
    std::unordered_map<std::string, std::size_t> by_name_;
    std::vector<std::string> strings_;
    std::unordered_map<std::string, std::uint32_t> string_ids_;
};

}  // namespace vecdb
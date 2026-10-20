/**
 * @file wal_ops.cpp
 * @brief Encoding and decoding logged operations; see wal_ops.h.
 */
#include "wal_ops.h"

#include <cmath>
#include <cstring>

#include "serialize.h"

namespace vecdb::walop {

namespace {

class Out {
public:
    template <class T>
    void put(T v) {
        unsigned char b[sizeof(T)];
        persist::to_little_endian(v, b);
        bytes.insert(bytes.end(), b, b + sizeof(T));
    }
    void put_string(std::string_view s) {
        put<std::uint32_t>(static_cast<std::uint32_t>(s.size()));
        bytes.insert(bytes.end(), s.begin(), s.end());
    }
    void put_floats(std::span<const float> v) {
        put<std::uint64_t>(v.size());
        for (float x : v) put<float>(x);
    }
    void put_metadata(const Metadata& m) {
        put<std::uint32_t>(static_cast<std::uint32_t>(m.fields().size()));
        for (const auto& [name, value] : m.fields()) {
            put_string(name);
            put<std::uint8_t>(value ? 1 : 0);  // 0: unset the field
            if (!value) continue;
            put<std::uint8_t>(static_cast<std::uint8_t>(value->type));
            switch (value->type) {  // GCOVR_EXCL_BR_LINE: every type has a case
                case FieldType::Int: put<std::int64_t>(value->i); break;
                case FieldType::Float: put<double>(value->f); break;
                case FieldType::Bool: put<std::uint8_t>(value->b ? 1 : 0); break;
                case FieldType::Keyword: put_string(value->s); break;
                case FieldType::Tags:
                    put<std::uint32_t>(static_cast<std::uint32_t>(value->tags.size()));
                    for (const auto& t : value->tags) put_string(t);
                    break;
            }
        }
    }
    std::vector<unsigned char> bytes;
};

class In {
public:
    explicit In(std::span<const unsigned char> b) : b_(b) {}
    template <class T>
    T get() {
        need(sizeof(T));
        const T v = persist::from_little_endian<T>(b_.data() + at_);
        at_ += sizeof(T);
        return v;
    }
    std::string get_string() {
        const std::uint32_t n = get<std::uint32_t>();
        need(n);
        std::string s(reinterpret_cast<const char*>(b_.data() + at_), n);
        at_ += n;
        return s;
    }
    std::vector<float> get_floats() {
        const std::uint64_t n = get<std::uint64_t>();
        if (n > (b_.size() - at_) / 4) throw FormatError("log record: vector longer than its record");
        std::vector<float> v(static_cast<std::size_t>(n));
        for (float& x : v) x = get<float>();
        return v;
    }
    Metadata get_metadata() {
        Metadata m;
        const std::uint32_t n = get<std::uint32_t>();
        for (std::uint32_t i = 0; i < n; ++i) {
            std::string name = get_string();
            if (get<std::uint8_t>() == 0) {
                m.unset(std::move(name));
                continue;
            }
            switch (static_cast<FieldType>(get<std::uint8_t>())) {
                case FieldType::Int: m.set(std::move(name), get<std::int64_t>()); break;
                case FieldType::Float: m.set(std::move(name), get<double>()); break;
                case FieldType::Bool: m.set(std::move(name), get<std::uint8_t>() != 0); break;
                case FieldType::Keyword: m.set(std::move(name), get_string()); break;
                case FieldType::Tags: {
                    const std::uint32_t k = get<std::uint32_t>();
                    if (k > b_.size() - at_) throw FormatError("log record: more tags than bytes");
                    std::vector<std::string> tags;
                    for (std::uint32_t t = 0; t < k; ++t) tags.push_back(get_string());
                    m.set_tags(std::move(name), std::move(tags));
                    break;
                }
                default: throw FormatError("log record: unknown metadata type");
            }
        }
        return m;
    }
    std::vector<std::uint64_t> get_ids() {
        const std::uint64_t n = get<std::uint64_t>();
        if (n > (b_.size() - at_) / 8) throw FormatError("log record: more IDs than bytes");
        std::vector<std::uint64_t> ids(static_cast<std::size_t>(n));
        for (auto& id : ids) id = get<std::uint64_t>();
        return ids;
    }
    void done() const {
        if (at_ != b_.size()) throw FormatError("log record: bytes left over");
    }

private:
    void need(std::size_t n) const {
        if (n > b_.size() - at_) throw FormatError("log record shorter than its contents");
    }
    std::span<const unsigned char> b_;
    std::size_t at_ = 0;
};

}  // namespace

std::vector<unsigned char> encode_add(std::uint64_t id, std::span<const float> vector, const Metadata& metadata) {
    Out o;
    o.put<std::uint64_t>(id);
    o.put_floats(vector);
    o.put_metadata(metadata);
    return std::move(o.bytes);
}

std::vector<unsigned char> encode_add_batch(std::span<const std::uint64_t> ids, std::span<const float> vectors,
                                            std::span<const Metadata> metadata) {
    Out o;
    o.put<std::uint64_t>(ids.size());
    for (std::uint64_t id : ids) o.put<std::uint64_t>(id);
    o.put_floats(vectors);
    o.put<std::uint8_t>(metadata.empty() ? 0 : 1);
    for (const Metadata& m : metadata) o.put_metadata(m);
    return std::move(o.bytes);
}

std::vector<unsigned char> encode_ids(std::span<const std::uint64_t> ids) {
    Out o;
    o.put<std::uint64_t>(ids.size());
    for (std::uint64_t id : ids) o.put<std::uint64_t>(id);
    return std::move(o.bytes);
}

std::vector<unsigned char> encode_update(std::uint64_t id, std::span<const float> vector, const Metadata* metadata) {
    Out o;
    o.put<std::uint64_t>(id);
    o.put_floats(vector);
    o.put<std::uint8_t>(metadata ? 1 : 0);
    if (metadata) o.put_metadata(*metadata);
    return std::move(o.bytes);
}

std::vector<unsigned char> encode_set_metadata(std::span<const std::uint64_t> ids, const Metadata& changes) {
    Out o;
    o.put<std::uint64_t>(ids.size());
    for (std::uint64_t id : ids) o.put<std::uint64_t>(id);
    o.put_metadata(changes);
    return std::move(o.bytes);
}

std::vector<unsigned char> encode_abort(std::uint64_t lsn) {
    Out o;
    o.put<std::uint64_t>(lsn);
    return std::move(o.bytes);
}

std::vector<unsigned char> encode_upkeep(const UpkeepOptions& u) {
    Out o;
    o.put<std::uint8_t>(u.consolidate ? 1 : 0);
    o.put<std::uint8_t>(u.reconnect ? 1 : 0);
    o.put<std::uint64_t>(u.max_nodes);
    return std::move(o.bytes);
}

std::vector<unsigned char> encode_rebuild(const RebuildOptions& r) {
    Out o;
    o.put<std::uint8_t>(r.retrain ? 1 : 0);
    return std::move(o.bytes);
}

std::vector<unsigned char> encode_field(std::string_view field) {
    Out o;
    o.put_string(field);
    return std::move(o.bytes);
}

std::vector<unsigned char> encode_planner(const PlannerParams& p) {
    Out o;
    o.put<std::uint64_t>(p.max_exact_matches);
    o.put<double>(p.max_exact_fraction);
    o.put<std::uint64_t>(p.sample_size);
    o.put<double>(p.max_ef_multiplier);
    return std::move(o.bytes);
}

std::vector<unsigned char> encode_fraction(double fraction) {
    Out o;
    o.put<double>(fraction);
    return std::move(o.bytes);
}

Op decode(std::uint8_t type, std::span<const unsigned char> payload) {
    In in(payload);
    Op op;
    op.type = static_cast<Type>(type);
    switch (type) {
        case kAdd: {
            op.ids.push_back(in.get<std::uint64_t>());
            op.vectors = in.get_floats();
            op.metadata.push_back(in.get_metadata());
            break;
        }
        case kAddBatch: {
            op.ids = in.get_ids();
            op.vectors = in.get_floats();
            if (in.get<std::uint8_t>() != 0)
                for (std::size_t i = 0; i < op.ids.size(); ++i) op.metadata.push_back(in.get_metadata());
            break;
        }
        case kRemove:
        case kRemoveBatch: op.ids = in.get_ids(); break;
        case kUpdate: {
            op.ids.push_back(in.get<std::uint64_t>());
            op.vectors = in.get_floats();
            if (in.get<std::uint8_t>() != 0) op.metadata.push_back(in.get_metadata());
            break;
        }
        case kSetMetadata:
        case kSetMetadataIds: {
            op.ids = in.get_ids();
            op.metadata.push_back(in.get_metadata());
            break;
        }
        case kAbort: op.lsn = in.get<std::uint64_t>(); break;
        case kCompact:
        case kTrainCodec:
        case kTrainLists: break;
        case kUpkeep: {
            op.upkeep.consolidate = in.get<std::uint8_t>() != 0;
            op.upkeep.reconnect = in.get<std::uint8_t>() != 0;
            op.upkeep.max_nodes = static_cast<std::size_t>(in.get<std::uint64_t>());
            break;
        }
        case kRebuild: op.rebuild.retrain = in.get<std::uint8_t>() != 0; break;
        case kCreatePayloadIndex: op.field = in.get_string(); break;
        case kSetPlanner: {
            op.planner.max_exact_matches = static_cast<std::size_t>(in.get<std::uint64_t>());
            op.planner.max_exact_fraction = in.get<double>();
            op.planner.sample_size = static_cast<std::size_t>(in.get<std::uint64_t>());
            op.planner.max_ef_multiplier = in.get<double>();
            break;
        }
        case kSetAutoCompact: op.fraction = in.get<double>(); break;
        default: throw FormatError("log record of unknown type " + std::to_string(type));
    }
    in.done();
    return op;
}

}  // namespace vecdb::walop
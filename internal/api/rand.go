package api

import "math/rand/v2"

// randUint64 supplies the fallback point for weighted selection when a request
// carries no entity key. It is a variable so tests can make routing
// deterministic without threading a source through every handler.
var randUint64 = rand.Uint64

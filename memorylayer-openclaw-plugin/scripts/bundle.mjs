// Bundle the plugin into a single ESM file for OpenClaw to load.
//
// Why bundle?
// - `@scitrera/memorylayer-sdk` and `@scitrera/aether-client` are not
//   published to npm; they're `file:` deps in dev. OpenClaw's
//   `plugins install` resolves runtime deps via npm, which can't see those
//   file: paths in the staging dir.
// - Bundling inlines the JS so the published plugin tarball declares only
//   the truly external runtime deps: @grpc/grpc-js + @grpc/proto-loader
//   (used by aether-client via native bindings; cannot be bundled).
//
// `openclaw` (and `openclaw/*` subpaths) are provided by the OpenClaw host
// at runtime — also external.

import { build } from "esbuild";

const result = await build({
  entryPoints: ["dist/src/index.js"],
  outfile: "dist/bundle/index.js",
  bundle: true,
  platform: "node",
  format: "esm",
  target: "node22",
  // Externals: anything we don't want inlined.
  //  - openclaw + subpaths: host-provided
  //  - @grpc/*: native bindings, can't be bundled
  //  - node: built-in modules (esbuild handles via platform=node but listing
  //    common ones makes intent explicit)
  external: [
    "openclaw",
    "openclaw/*",
    "@openclaw/*",
    "@grpc/grpc-js",
    "@grpc/proto-loader",
  ],
  logLevel: "info",
  metafile: true,
});

// Tiny report so build output makes obvious what landed in the bundle.
const inputs = Object.keys(result.metafile.inputs).filter(
  (p) => !p.startsWith("node_modules/openclaw"),
);
const bundled = inputs.filter((p) => p.startsWith("node_modules/"));
console.log(
  `[bundle] entries: ${inputs.length - bundled.length} app + ${bundled.length} bundled deps`,
);

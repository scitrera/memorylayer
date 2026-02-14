import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/ml/:path*",
        destination: `${process.env.MEMORYLAYER_URL || "http://localhost:61001"}/:path*`,
      },
    ];
  },
};

export default nextConfig;

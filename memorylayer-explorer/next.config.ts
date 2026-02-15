import type {NextConfig} from "next";

const nextConfig: NextConfig = {
    output: "standalone",
    allowedDevOrigins: ["localhost", "spark-2918",],
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

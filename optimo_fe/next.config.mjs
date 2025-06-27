import path from "path";

const nextConfig = {
  webpack(config) {
    config.resolve.alias["@"] = path.resolve("./src");
    return config;
  },
  reactStrictMode: false,
};

export default nextConfig;

import {defineConfig} from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
    site: 'https://docs.memorylayer.ai',
    server: {
        port: 4322,
        host: true, // Equivalent to --host flag
    },
    integrations: [
        starlight({
            title: 'memorylayer.ai Documentation',
            description: 'Memory infrastructure for LLM-powered agents',
            logo: {
                light: './src/assets/logo-light.svg',
                dark: './src/assets/logo-dark.svg',
                replacesTitle: false,
            },
            social: {
                github: 'https://github.com/scitrera/memorylayer',
            },
            editLink: {
                baseUrl: 'https://github.com/scitrera/memorylayer/edit/main/oss/docs/',
            },
            customCss: [
                './src/styles/custom.css',
            ],
            components: {
                Footer: './src/components/Footer.astro',
            },
            sidebar: [
                {
                    label: 'Getting Started',
                    autogenerate: {directory: 'getting-started'},
                },
                {
                    label: 'Server',
                    autogenerate: {directory: 'server'},
                },
                {
                    label: 'Python SDK',
                    autogenerate: {directory: 'sdk-python'},
                },
                {
                    label: 'TypeScript SDK',
                    autogenerate: {directory: 'sdk-typescript'},
                },
                {
                    label: 'Integrations',
                    autogenerate: {directory: 'integrations'},
                },
                {
                    label: 'Guides',
                    autogenerate: {directory: 'guides'},
                },
                {
                    label: 'Reference',
                    autogenerate: {directory: 'reference'},
                },
            ],
        }),
    ],
    vite: {
        server: {
            allowedHosts: ["spark-2918", "localhost"],
        }
    }
});

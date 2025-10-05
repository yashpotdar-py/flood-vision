module.exports = {
  branches: ['main'],
  repositoryUrl: 'https://github.com/btech-project-ai/flood-vision',
  plugins: [
    '@semantic-release/commit-analyzer',
    '@semantic-release/release-notes-generator',
    ['@semantic-release/changelog', {
      changelogFile: 'CHANGELOG.md'
    }],
    ['@semantic-release/git', {
      assets: ['CHANGELOG.md'],
      message: 'docs: :memo: update changelog for ${nextRelease.version} [skip ci]'
    }],
    '@semantic-release/github'
  ],
};

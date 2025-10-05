module.exports = {
  branches: ['main'],
  repositoryUrl: 'https://github.com/yashpotdar-py/flood-vision.git',
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

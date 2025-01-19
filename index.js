import fs from 'fs';

(async () => {
    const data = fs.readFileSync('data.txt', 'utf8');
    console.log(data);
})();

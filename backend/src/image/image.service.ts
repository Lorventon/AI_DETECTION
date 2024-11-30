import { Injectable } from '@nestjs/common';
import { PythonShell, Options } from 'python-shell';
import * as fs from 'fs';
import * as path from 'path';

@Injectable()
export class ImageService {
  public async processImage(file: {
    buffer: Buffer;
    originalname: string;
  }): Promise<any> {
    const tempFilePath = path.join(
      __dirname,
      'temp_image_' + Date.now() + path.extname(file.originalname),
    );

    try {
      fs.writeFileSync(tempFilePath, file.buffer);

      const result = await this.runPythonScript(tempFilePath);

      return result;
    } catch (error) {
      console.error('Ошибка при обработке изображения:', error);
      throw error;
    } finally {
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
      }
    }
  }

  private async runPythonScript(imagePath: string): Promise<any> {
    return new Promise((resolve) => {
      const options: Options = {
        mode: 'text',
        pythonOptions: ['-u'],
        args: [imagePath],
        scriptPath: 'D:\\Projects\\HackInHome2024\\test',
      };

      PythonShell.run('main.py', options).then((messages) => {
        const parsedResult = JSON.parse(messages[0]);
        resolve(parsedResult);
      });
    });
  }
}

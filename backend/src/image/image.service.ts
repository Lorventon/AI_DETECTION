import { Injectable } from '@nestjs/common';
import { PythonShell, Options } from 'python-shell';
import * as fs from 'fs';
import * as path from 'path';

@Injectable()
export class ImageService {
  public async processImage(file: {
    buffer: Buffer;
    originalname: string;
  }): Promise<{ imageBase64: string; detections: string[] }> {
    const tempFilePath = path.join(
      __dirname,
      'temp_image_' + Date.now() + path.extname(file.originalname),
    );

    const outputFilePath = path.join(
      __dirname,
      'processed_image_' + Date.now() + path.extname(file.originalname),
    );

    try {
      fs.writeFileSync(tempFilePath, file.buffer);
      const detectionResults = await this.runPythonScript(
        tempFilePath,
        outputFilePath,
      );

      const processedImageBuffer = fs.readFileSync(outputFilePath);
      const imageBase64 = `data:image/${path.extname(outputFilePath).slice(1)};base64,${processedImageBuffer.toString('base64')}`;

      // Возврат как Base64 изображения, так и результатов детекции
      return {
        imageBase64,
        detections: detectionResults,
      };
    } catch (error) {
      console.error('Ошибка при обработке изображения:', error);
      throw error;
    } finally {
      if (fs.existsSync(tempFilePath)) {
        fs.unlinkSync(tempFilePath);
      }
      if (fs.existsSync(outputFilePath)) {
        fs.unlinkSync(outputFilePath);
      }
    }
  }

  private async runPythonScript(
    imagePath: string,
    outputPath: string,
  ): Promise<any> {
    return new Promise((resolve) => {
      const options: Options = {
        mode: 'text',
        pythonOptions: ['-u'],
        args: [imagePath, outputPath],
        scriptPath: 'D:\\Projects\\HackInHome2024\\model',
      };

      PythonShell.run('test.py', options).then((messages) => {
        console.log(messages);
      });
    });
  }
}

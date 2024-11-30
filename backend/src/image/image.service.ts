import { Injectable } from '@nestjs/common';

@Injectable()
export class ImageService {
  public setImage(file: Express.Multer.File): void {
    console.log('Отправенный файл: ', file);
  }
}
